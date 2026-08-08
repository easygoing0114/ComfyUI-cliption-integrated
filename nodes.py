import logging
import os
import re
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

import comfy.model_management


def _fix_punctuation_spacing(text: str) -> str:
    """Remove stray whitespace before commas and periods left over from
    CLIPTokenizer's BPE decoding (e.g. "cats , dogs ." -> "cats, dogs.")."""
    return re.sub(r"\s+([,.])", r"\1", text)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, self_attn_mask: torch.Tensor):
        # self attention with mask
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=x, key=x, value=x, attn_mask=self_attn_mask, need_weights=False, is_causal=True
        )
        x = residual + attn_output

        # cross attention
        residual = x
        x = self.norm2(x)
        attn_output, _ = self.cross_attn(query=x, key=memory, value=memory, need_weights=False)
        x = residual + attn_output

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.mlp(x)

        return x


class Captioner(nn.Module):
    def __init__(self, config, vision_embed_dim: int, vocab_size: int):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.max_length = config.max_length

        # projection from ViT dimension to decoder dimension
        self.projection = nn.Linear(vision_embed_dim, self.hidden_dim)
        self.memory_pos_embedding = nn.Parameter(torch.zeros(1, 257, self.hidden_dim))

        # decoder layers
        self.layers = nn.ModuleList(
            [DecoderBlock(config.hidden_dim, config.num_heads) for _ in range(config.num_blocks)]
        )

        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_length)
        self.register_buffer("causal_mask", causal_mask, persistent=False)


class CLIPtionModel(nn.Module):
    def __init__(self, config, clip, clip_vision, device=None):
        super().__init__()

        if not hasattr(clip, "cond_stage_model"):
            raise ValueError("CLIP is missing from model checkpoint")
        if not hasattr(clip.cond_stage_model, "clip_l"):
            raise ValueError("Must use model which includes CLIP-L")

        # store CLIP model references
        self.clip_text = clip
        self.clip_vision = clip_vision
        # use specified device, or fall back to ComfyUI default at runtime
        self.inference_device = device
        self.tokenizer = clip.tokenizer.clip_l.tokenizer
        self.text_model = clip.cond_stage_model.clip_l.transformer.text_model

        # clip.cond_stage_model.clip_l.transformer.text_projection is empty
        # so load a copy from the CLIPtion safetensors file instead
        self.text_projection = nn.Linear(768, 768, bias=False)

        # create caption decoder
        self.captioner = Captioner(config, 1024, self.tokenizer.vocab_size)

        # use CLIP's token embeddings for output projection
        clip_embed_weight = self.text_model.embeddings.token_embedding.weight
        self.output_projection = nn.Linear(
            self.captioner.hidden_dim, self.tokenizer.vocab_size, bias=False
        )
        self.output_projection.weight = nn.Parameter(clip_embed_weight.clone())

    def generate_beam(self, images: torch.Tensor, beam_width: int = 4) -> list:
        device = self.inference_device or comfy.model_management.get_torch_device()
        image_features, image_embeds = self._images_to_embeds(images, device)

        captions = []
        for image_idx in range(image_features.size(0)):
            features = image_features[image_idx].unsqueeze(0)
            candidates = self._beam_search(
                features, image_embeds[image_idx : image_idx + 1], device, beam_width
            )

            # pick highest scoring candidate
            candidates.sort(key=lambda x: x[0])
            for score, text in candidates:
                logging.debug(f"({score:.3f}) {text}")
            captions.append(candidates[-1][1])
        return captions

    def _beam_search(
        self,
        image_features: torch.Tensor,
        image_embed: torch.Tensor,
        device: torch.device,
        beam_width: int,
    ):
        tokenizer = self.tokenizer
        captioner = self.captioner
        token_embedding = self.text_model.embeddings.token_embedding
        pos_embedding = self.text_model.embeddings.position_embedding
        vocab_size = tokenizer.vocab_size

        # project image features
        # determine dtype from captioner weights to handle FP32 models correctly
        model_dtype = next(captioner.parameters()).dtype
        memory = captioner.projection(image_features.to(dtype=model_dtype))
        memory = memory + captioner.memory_pos_embedding

        # start with beam_width copies of BOS token
        current_tokens = torch.full(
            (beam_width, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        scores = torch.zeros(beam_width, device=device)

        for step in range(captioner.max_length - 2):
            # embed current tokens
            token_embeddings = token_embedding(current_tokens)
            positions = torch.arange(current_tokens.size(1), device=device)
            pos_embeddings = pos_embedding(positions)
            # cast to model dtype to handle FP32/FP16 mismatch
            x = (token_embeddings + pos_embeddings).to(dtype=model_dtype)

            # run decoder layers
            seq_len = x.size(1)
            mask = captioner.causal_mask[:seq_len, :seq_len]
            for layer in captioner.layers:
                x = layer(x, memory.repeat(beam_width, 1, 1), self_attn_mask=mask)

            # get next token log probabilities
            logits = self.output_projection(x[:, -1:])
            log_probs = F.log_softmax(logits, dim=-1)

            if step == 0:
                # pick top-k tokens for first step
                scores = log_probs.squeeze(1)[0]
                scores, indices = scores.topk(beam_width)
                current_tokens = torch.cat(
                    [current_tokens[0:1].repeat(beam_width, 1), indices.unsqueeze(1)], dim=1
                )
            else:
                # calculate scores for next tokens [beam_width x vocab_size]
                next_scores = scores.unsqueeze(1) + log_probs.squeeze(1)

                # force sequences to continue EOS after first one
                prev_is_eos = current_tokens[:, -1] == tokenizer.eos_token_id
                vocab_mask = torch.zeros_like(next_scores)
                vocab_mask[prev_is_eos] = float("-inf")
                vocab_mask[prev_is_eos, tokenizer.eos_token_id] = 0
                next_scores = next_scores + vocab_mask

                # pick top beam_width sequences
                next_scores = next_scores.view(-1)
                scores, indices = next_scores.topk(beam_width)
                beam_indices = indices // vocab_size  # which sequence each came from
                token_indices = indices % vocab_size  # which token to append

                current_tokens = torch.cat(
                    [current_tokens[beam_indices], token_indices.unsqueeze(1)], dim=1
                )

            # check if all beams ended with EOS
            if (current_tokens[:, -1] == tokenizer.eos_token_id).all():
                break

        # add final EOS token
        current_tokens = torch.cat(
            [current_tokens, torch.full((beam_width, 1), tokenizer.eos_token_id, device=device)],
            dim=1,
        )

        # rank final candidates by CLIP similarity
        candidates = []
        for idx in range(beam_width):
            tokens = current_tokens[idx]
            # trim everything after the first EOS token (inclusive) before decoding
            eos_positions = (tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                tokens = tokens[: eos_positions[0]]
            text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            text = _fix_punctuation_spacing(text)
            text_embeds = self._text_to_embed(text, device)
            clip_sim = torch.sum(image_embed * text_embeds, dim=-1)[0]
            candidates.append((clip_sim.item(), text))
        return candidates

    def _images_to_embeds(self, images: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.size(-1) == 1:
            images = images.repeat(1, 1, 1, 3)
        elif images.size(-1) == 4:
            images = images[..., :3]

        outputs = self.clip_vision.encode_image(images)
        # features go into the FP16 CLIPtion decoder, so cast to FP16
        features = outputs.last_hidden_state.to(device, dtype=torch.float16)
        if features.size(2) != 1024:
            raise ValueError(
                f"Expected image features to have 1024 dimensions but got {features.size(2)}. Please ensure you are using CLIP L."
            )

        # embeds are used only for CLIP similarity scoring, preserve original dtype (FP32) for accuracy
        embeds = outputs.image_embeds.to(device)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return features, embeds

    def _text_to_embed(self, text: str, device: torch.device) -> torch.Tensor:
        # load CLIP model and disable final projection since that's missing from comfy checkpoints
        self.clip_text.load_model()
        self.clip_text.cond_stage_model.reset_clip_options()
        self.clip_text.cond_stage_model.set_clip_options({"projected_pooled": False})

        # calculate text embedding
        tokens = self.clip_text.tokenize(text)
        clip_l = self.clip_text.cond_stage_model.clip_l
        _, pooled = clip_l.encode_token_weights(tokens["l"])
        # preserve original dtype (FP32) for accurate CLIP similarity scoring
        text_embeds = self.text_projection(pooled.to(device))
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds


class CLIPtionBeamSearchIntegrated:
    """
    Combined CLIPtion loader + beam search node.
    Loads the CLIPtion decoder on demand at caption time rather than at
    workflow-load time, and can optionally unload it from VRAM afterwards.
    The CLIPtion safetensors weights are kept on CPU between runs so that
    re-loading does not require a disk read.
    """

    CATEGORY = "pharmapsychotic"
    FUNCTION = "caption"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)

    # CLIPtion decoder config (fixed for the released checkpoint)
    _CAPTIONER_CONFIG = SimpleNamespace(hidden_dim=768, num_heads=8, num_blocks=6, max_length=77)
    _SAFETENSORS_FILE = "CLIPtion_20241219_fp16.safetensors"
    _HF_REPO_ID = "pharmapsychotic/CLIPtion"
    _HF_REVISION = "15ee8cb77a902616478a033332011ff640e72277"

    def __init__(self):
        self._model: Optional[CLIPtionModel] = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP text encoder (must include CLIP-L)."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "CLIP vision encoder (must be CLIP-L)."}),
                "image": ("IMAGE",),
                "beam_width": (
                    "INT",
                    {"default": 4, "min": 1, "max": 64, "tooltip": "Number of beams to maintain during search."},
                ),
                "unload_after_run": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Unload CLIPtion decoder from VRAM after captioning."},
                ),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    def caption(self, clip, clip_vision, image: torch.Tensor, beam_width: int = 4,
                unload_after_run: bool = False, device: str = "default"):
        try:
            self._load_model(clip, clip_vision, device)
            with torch.inference_mode():
                captions = self._model.generate_beam(image, beam_width)
        finally:
            if unload_after_run:
                self._unload_model()

        return (captions,)

    def _load_model(self, clip, clip_vision, device: str):
        """Load CLIPtion decoder from disk and move to target device."""
        if self._model is not None:
            return

        base_path = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_path, self._SAFETENSORS_FILE)
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = hf_hub_download(
                repo_id=self._HF_REPO_ID, filename=self._SAFETENSORS_FILE, revision=self._HF_REVISION
            )

        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        tp_dict = {"weight": state_dict.pop("text_projection.weight")}

        inference_device = torch.device("cpu") if device == "cpu" else None
        model = CLIPtionModel(self._CAPTIONER_CONFIG, clip, clip_vision, device=inference_device)
        model.captioner.load_state_dict(state_dict)
        model.text_projection.load_state_dict(tp_dict)
        model.eval()

        load_device = inference_device or comfy.model_management.get_torch_device()
        model.to(load_device, dtype=torch.float16)
        # keep text_projection in FP32 to preserve CLIP similarity scoring accuracy
        model.text_projection.to(dtype=torch.float32)

        self._model = model
        logging.info(f"CLIPtionBeamSearchIntegrated: decoder loaded on {load_device}")

    def _unload_model(self):
        """Remove the CLIPtion decoder from VRAM and CPU RAM completely."""
        if self._model is None:
            return
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("CLIPtionBeamSearchIntegrated: decoder unloaded")


NODE_CLASS_MAPPINGS = {
    "CLIPtionBeamSearchIntegrated": CLIPtionBeamSearchIntegrated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPtionBeamSearchIntegrated": "CLIPtion Beam Search (Integrated)",
}
