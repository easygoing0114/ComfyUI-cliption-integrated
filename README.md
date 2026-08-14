<div align="center">
<img width="800" height="343" alt="ComfyUI Cliption Integrated Nodes thumbnail" src="images/cliption_banner_image.png">
</div>

# ComfyUI-cliption-integrated

A streamlined fork of [pharmapsychotic/comfy-cliption](https://github.com/pharmapsychotic/comfy-cliption) for ComfyUI, providing a single, integrated **CLIPtion Beam Search** node.

[CLIPtion](https://huggingface.co/pharmapsychotic/CLIPtion) is a fast, lightweight image captioning model built on top of CLIP. This fork updates it for ComfyUI's V3 node scheme and consolidates the node pack into a single, integrated node. Processing has been sped up for GPU inference, and a VRAM offload option has been added.

## CLIPtion Beam Search (Integrated) Node

### Single Image

<div align="center">
<img width="800" height="468" alt="single image sample workflow" src="images/comfyui-cliption-integrated_sample_workflow_20260814.png">
</div>

### Multiple Images

<div align="center">
<img width="800" height="351" alt="multiple images sample workflow" src="images/comfyui-cliption-integrated_multipul_images_sample_workflow_20260814.png">
</div>

Loads the CLIPtion decoder on demand at caption time (rather than at workflow-load time) and runs beam search to caption an image, ranking the resulting candidates by CLIP similarity to the input image.

### Inputs

| Name | Type | Description |
| --- | --- | --- |
| `clip` | `CLIP` | CLIP text encoder (must include CLIP-L). |
| `clip_vision` | `CLIP_VISION` | CLIP vision encoder (must be CLIP-L). |
| `image` | `IMAGE` | Input Image (s). |

### Settings

| Name | Type | Description |
| --- | --- | --- |
| `beam_width` | `INT` | Number of beams to maintain during search (default: 4). |
| `unload_after_run` | `BOOLEAN` | Unload the CLIPtion decoder from VRAM after captioning (default: False). |
| `device` (optional) | `["default", "cpu"]` | Inference device override. |

### Output

| Name | Type | Description |
| --- | --- | --- |
| `STRING` | `STRING` (list) | Generated caption (s). |

## CLIPtion Model

The CLIPtion decoder weights (`CLIPtion_20241219_fp16.safetensors`) are downloaded automatically from [easygoing0114/ComfyUI-use-models](https://huggingface.co/easygoing0114/ComfyUI-use-models/tree/main) on Hugging Face the first time the node runs, and are saved to `ComfyUI/models/cliption` for reuse on subsequent runs. You can also place a copy of the file in that folder yourself ahead of time to skip the automatic download.

## Recommended CLIP-SAE-ViT-L-14 Model

For the `clip` and `clip_vision` inputs, we recommend [zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14), a free, high-accuracy fine-tune of CLIP ViT-L/14 released by zer0int. This model has also been mirrored to the [easygoing0114/ComfyUI-use-models](https://huggingface.co/easygoing0114/ComfyUI-use-models/tree/main) repository above.

Download `CLIP-SAE-ViT-L-14_FP32.safetensors` and place it in **both** of the following folders:

- `ComfyUI/models/text_encoders`
- `ComfyUI/models/clip_vision`

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/easygoing0114/ComfyUI-cliption-integrated.git
```

Restart ComfyUI. The node should now appear in the node search as **CLIPtion Beam Search (Integrated)**.

## Credits

- [pharmapsychotic/comfy-cliption](https://github.com/pharmapsychotic/comfy-cliption) — original ComfyUI node pack this fork is based on.
- [pharmapsychotic/CLIPtion](https://huggingface.co/pharmapsychotic/CLIPtion) — the underlying CLIPtion model.
- [zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14) — recommended high-accuracy CLIP-L model, freely released by zer0int.

## License

This project is licensed under the MIT License.