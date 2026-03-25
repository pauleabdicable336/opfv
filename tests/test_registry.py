import pytest
from opfv.registry import PhiRegistryError, registered_phi_names, resolve_phi_pair


def test_registered_names() -> None:
    assert "fourier" in registered_phi_names()


def test_resolve_fourier() -> None:
    s, v = resolve_phi_pair(["fourier"])
    assert len(s) == 1 and len(v) == 1


def test_unknown_phi() -> None:
    with pytest.raises(PhiRegistryError):
        resolve_phi_pair(["not_a_real_phi"])
