"""Infrastructure adapters and health contracts."""

from .health import DependencyCheck, DependencyHealthRegistry

__all__ = ["DependencyCheck", "DependencyHealthRegistry"]
