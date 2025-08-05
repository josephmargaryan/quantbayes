# ── Registry storage ────────────────────────────────────────────────────
_SEG_MODELS: dict[str, callable] = {}
_CLS_MODELS: dict[str, callable] = {}


def register_seg_model(name: str):
    def decorator(fn: callable):
        _SEG_MODELS[name] = fn
        return fn

    return decorator


def register_cls_model(name: str):
    def decorator(fn: callable):
        _CLS_MODELS[name] = fn
        return fn

    return decorator


def get_seg_model(name: str):
    return _SEG_MODELS[name]


def get_cls_model(name: str):
    return _CLS_MODELS[name]


def list_seg_models() -> list[str]:
    return sorted(_SEG_MODELS.keys())


def list_cls_models() -> list[str]:
    return sorted(_CLS_MODELS.keys())


# ── Now that the above exists, import the modules that register models ──
# Replace these with your actual package path.
import quantbayes.torch_based.vision.models.segmentation  # noqa: F401
import quantbayes.torch_based.vision.models.classification  # noqa: F401
