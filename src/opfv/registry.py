# Modifications Copyright (c) 2026 Tatsuhiro Shimizu

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from opfv.domain import estimators as est


class PhiRegistryError(LookupError):
    """Unknown phi feature name in configuration."""


def resolve_phi_pair(
    names: Sequence[str],
) -> tuple[list[Callable[..., object]], list[Callable[..., object]]]:
    """Map YAML names to (scalar, vector) function pairs for Prognosticator."""
    scalars: list[Callable[..., object]] = []
    vectors: list[Callable[..., object]] = []
    for name in names:
        s, v = _SINGLETON.lookup_pair(name)
        scalars.append(s)
        vectors.append(v)
    return scalars, vectors


class _PhiRegistry:
    """Registers time-feature function pairs used in synthetic experiments."""

    def __init__(self) -> None:
        self._pairs: dict[str, tuple[Callable[..., object], Callable[..., object]]] = {
            "fourier": (est.fourier_scalar, est.fourier_vec),
            "exponential": (est.exponential_scalar, est.exponential_vec),
        }

    def lookup_pair(self, name: str) -> tuple[Callable[..., object], Callable[..., object]]:
        key = name.strip().lower()
        if key not in self._pairs:
            raise PhiRegistryError(f"Unknown phi pair {name!r}. Known: {sorted(self._pairs)}")
        return self._pairs[key]


_SINGLETON = _PhiRegistry()


def registered_phi_names() -> list[str]:
    return sorted(_SINGLETON._pairs.keys())


def resolve_phi_pairs_from_config(
    phi_pair_names: Iterable[str],
) -> tuple[list[Callable[..., object]], list[Callable[..., object]]]:
    return resolve_phi_pair(list(phi_pair_names))
