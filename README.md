<div align="center">
<img width="800" height="343" alt="ComfyUI Cliption Integrated Nodes thumbnail" src="images/cliption_banner_image.png">
</div>

# ComfyUI-cliption-integrated

A streamlined fork of [pharmapsychotic/comfy-cliption](https://github.com/pharmapsychotic/comfy-cliption) for ComfyUI, providing a single, integrated **CLIPtion Beam Search** node.

[CLIPtion](https://github.com/pharmapsychotic/CLIPtion) is a fast, lightweight image captioning model built on top of CLIP. This fork simplifies the original node pack down to the beam-search workflow only.

## Node

### CLIPtion Beam Search (Integrated)

<div align="center">
<img width="800" height="468" alt="mask refine node sample" src="images/comfyui-cliption-integrated_sample_workflow_20260813.png">
</div>

Loads the CLIPtion decoder on demand at caption time (rather than at workflow-load time) and runs beam search to caption an image, ranking the resulting candidates by CLIP similarity to the input image.

**Inputs**

| Name | Type | Description |
| --- | --- | --- |
| `clip` | `CLIP` | CLIP text encoder (must include CLIP-L). |
| `clip_vision` | `CLIP_VISION` | CLIP vision encoder (must be CLIP-L). |
| `image` | `IMAGE` | Inoput Image to caption. |
| `beam_width` | `INT` | Number of beams to maintain during search (default: 4). |
| `unload_after_run` | `BOOLEAN` | Unload the CLIPtion decoder from VRAM after captioning (default: False). |
| `device` (optional) | `["default", "cpu"]` | Inference device override. |

**Output**

| Name | Type | Description |
| --- | --- | --- |
| `STRING` | `STRING` (list) | One caption per input image. |

The CLIPtion decoder weights (`CLIPtion_20241219_fp16.safetensors`) are downloaded automatically from [easygoing0114/ComfyUI-use-models](images/cliption_banner_image.png) on Hugging Face the first time the node runs, and are cached on disk afterwards.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/easygoing0114/ComfyUI-cliption-integrated.git
```

Restart ComfyUI. The node should now appear in the node search as **CLIPtion Beam Search (Integrated)**.

## Credits

- [pharmapsychotic/comfy-cliption](https://github.com/pharmapsychotic/comfy-cliption) — original ComfyUI node pack this fork is based on.
- [pharmapsychotic/CLIPtion](https://github.com/pharmapsychotic/CLIPtion) — the underlying CLIPtion model.

## License

This project is licensed under the MIT License.
