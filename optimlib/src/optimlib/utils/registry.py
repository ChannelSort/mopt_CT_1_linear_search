"""Explicit registries for YAML-to-class mapping."""

from __future__ import annotations

import re
from typing import Any, TypeVar

from optimlib.exceptions import ConfigurationError


T = TypeVar("T")


class Registry:
    """Registry for optimizers and objective functions."""

    def __init__(self) -> None:
        """Initialize empty maps."""
        self._optimizers: dict[str, type[Any]] = {}
        self._functions: dict[str, type[Any]] = {}

    def register_optimizer(self, name: str, cls: type[Any]) -> None:
        """Register optimizer class by explicit name."""
        self._register(self._optimizers, name, cls, kind="optimizer")

    def register_function(self, name: str, cls: type[Any]) -> None:
        """Register function class by explicit name."""
        self._register(self._functions, name, cls, kind="function")

    def get_optimizer_class(self, name: str) -> type[Any]:
        """Return registered optimizer class."""
        try:
            return self._optimizers[self._normalize(name)]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown optimizer: {name}") from exc

    def get_function_class(self, name: str) -> type[Any]:
        """Return registered function class."""
        try:
            return self._functions[self._normalize(name)]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown function: {name}") from exc

    def get_optimizer(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered optimizer."""
        return self.get_optimizer_class(name)(**kwargs)

    def get_function(self, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered function."""
        return self.get_function_class(name)(**kwargs)

    def optimizer_names(self) -> tuple[str, ...]:
        """Return registered optimizer names."""
        return tuple(sorted(self._optimizers))

    def function_names(self) -> tuple[str, ...]:
        """Return registered function names."""
        return tuple(sorted(self._functions))

    @classmethod
    def _normalize(cls, name: str) -> str:
        """Normalize registry keys while keeping YAML mapping explicit."""
        stripped = name.strip()
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", stripped).lower()
        return snake.replace("-", "_").replace(" ", "_")

    @classmethod
    def _register(cls, target: dict[str, type[Any]], name: str, item: type[Any], *, kind: str) -> None:
        key = cls._normalize(name)
        existing = target.get(key)
        if existing is not None and existing is not item:
            raise ConfigurationError(f"Conflicting {kind} registration for {name!r}.")
        target[key] = item


GLOBAL_REGISTRY = Registry()


def register_optimizer(name: str, cls: type[Any]) -> None:
    """Register optimizer in the process-global registry."""
    GLOBAL_REGISTRY.register_optimizer(name, cls)


def register_function(name: str, cls: type[Any]) -> None:
    """Register function in the process-global registry."""
    GLOBAL_REGISTRY.register_function(name, cls)
