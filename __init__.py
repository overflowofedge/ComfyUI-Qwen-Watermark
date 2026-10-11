try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError as exc:
    if exc.name not in {"comfy_execution", "comfy_execution.graph"}:
        raise
    # Repository tooling may inspect the package outside a ComfyUI runtime.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.3.0"
