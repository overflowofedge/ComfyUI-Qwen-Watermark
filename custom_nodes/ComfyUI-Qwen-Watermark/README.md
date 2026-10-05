# ComfyUI-Qwen-Watermark

Custom nodes used by the Qwen Image 2.1 watermark-removal workflow.

## Nodes

- `QWMDetect`: one Qwen3-VL inference with general and Logo-recovery scan instructions, automatic type classification, manual boxes, or a painted mask.
- `QWMResidualDetect`: post-edit residual check for watermark fragments, translucent ghosts, and seams.
- `QWMRefineMask`: SAM contour refinement and adaptive edge/shadow expansion.
- `QWMPrepare`: local crop, padding, and optional OpenCV pre-inpainting.
- `QWMComposite`: alignment, inward feathering, and protected paste-back.

## Installation

Copy this directory to `ComfyUI/custom_nodes/ComfyUI-Qwen-Watermark`, then install:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-Qwen-Watermark/requirements.txt
```

Restart ComfyUI after installation. The workflow also needs current ComfyUI core Qwen Image 2.1 nodes and the model files listed in the release package README.

`manual_mask` does not invoke SAM. It applies the node's base and adaptive dilation directly to the painted selection so borders, translucent edges, and shadows are included.
