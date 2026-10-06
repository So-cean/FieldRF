"""FieldRF models, with lazy exports so losses do not load network dependencies."""

from importlib import import_module

__all__ = ["FieldRFModel", "EndpointAligner"]


def __getattr__(name):
    modules = {"FieldRFModel": ".model", "EndpointAligner": ".endpoint_aligner"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
