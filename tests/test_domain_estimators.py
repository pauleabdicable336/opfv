import numpy as np
from opfv.domain.estimators import fourier_scalar, fourier_vec


def test_fourier_scalar_shape() -> None:
    x = np.array(0.5)
    out = fourier_scalar(x, n=2, K_plus_delta=8)
    assert out.shape[0] == 1 + 2 * 2


def test_fourier_vec_rows() -> None:
    x = np.array([0.0, 0.25])
    m = fourier_vec(x, n=1, K_plus_delta=4)
    assert m.shape[0] == 2
    assert m.shape[1] == 1 + 2
