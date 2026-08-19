"""Static frequency-domain chromatic-dispersion compensation (CDC)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cd_compensate(
    sig: NDArray[np.complex128],
    sample_rate: float,
    beta2: float,
    length_km: float,
    alpha_db_km: float = 0.0,
) -> NDArray[np.complex128]:
    """Apply the inverse linear channel transfer function in the frequency domain.

    .. math:: H^{-1}_{\\text{CD}}(\\omega) =
                  \\exp\\bigl(+\\alpha L/2 - j\\,\\beta_2\\omega^2 L/2\\bigr)

    The whole block is transformed once (fixed channel, so the filter is static).
    If ``alpha_db_km`` is non-zero the accumulated loss is *restored* as an
    ideal DSP gain stage (commonly pinned inside the equalizer instead).
    """
    n = len(sig)
    freq = np.fft.fftfreq(n, d=1.0 / sample_rate)
    omega = 2.0 * np.pi * freq

    alpha_lin = (alpha_db_km / 10.0 * np.log(10.0)) / 1e3
    length_m = length_km * 1e3

    phase = -1j * (beta2 / 2.0) * omega**2 * length_m
    if alpha_db_km != 0.0:
        phase = phase + (alpha_lin / 2.0) * length_m

    h_inv = np.exp(phase)
    return np.fft.ifft(np.fft.fft(sig) * h_inv)
