# Request-scoped storage for SHAP explanation factors.
# Uses ContextVar so each async request gets its own isolated copy.

from contextvars import ContextVar

_factors_var: ContextVar[list] = ContextVar("explanation_factors", default=[])
_method_var: ContextVar[str] = ContextVar("explainability_method", default="")


def set_factors(factors: list) -> None:
    _factors_var.set(factors)


def get_factors() -> list:
    return _factors_var.get()


def set_method(method: str) -> None:
    _method_var.set(method)


def get_method() -> str:
    return _method_var.get()


def clear() -> None:
    _factors_var.set([])
    _method_var.set("")
