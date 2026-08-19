"""CDC: fast chromatic-dispersion compensation reverses the linear channel."""

from __future__ import annotations

import numpy as np
from optical_dsp.dsp.cdc import cd_compensate
from optical_dsp.physics.channel import apply_analytical_linear_channel
from optical_dsp.physics.transmitter import rrc_taps
from optical_dsp.utils import QPSK, FibreParams


def test_cdc_inverts_linear_channel() -> None:
    sps, n_symbols = 4, 2048
    fibre = FibreParams()  # real SMF values, keeps NLSE gamma off here
    fs = 32e9 * sps

    rng = np.random.default_rng(0)
    sym = QPSK().symbols[rng.integers(0, 4, n_symbols)]
    up = np.zeros(n_symbols * sps, dtype=np.complex128)
    up[::sps] = sym
    h = rrc_taps(0.2, sps, 33)
    conv = np.convolve(up, h)
    sig = conv[(len(h) - 1) // 2 :][: n_symbols * sps]

    propagated = apply_analytical_linear_channel(sig, fs, 80.0, fibre)
    equalized = cd_compensate(propagated, fs, fibre.beta2, 80.0, fibre.alpha_db_km)
    # phase is perfectly recovered; tiny numerical mismatch at frame edges
    err = np.linalg.norm(equalized - sig) / np.linalg.norm(sig)
    assert err < 1e-6


def test_cdc_preserves_phase_rotation() -> None:
    n = 8192
    beta2 = -21.3e-27
    freq = np.fft.fftfreq(n, 1 / (32e9 * 4))
    omega = 2 * np.pi * freq
    delayer = np.exp(1j * (beta2 / 2.0) * omega**2 * 80e3)
    sig = np.fft.ifft(np.fft.fft(np.exp(1j * 2 * np.pi * 3e9 * np.arange(n) / (32e9 * 4))))
    delayed = np.fft.ifft(np.fft.fft(sig) * delayer)
    out = cd_compensate(delayed, 32e9 * 4, beta2, 80.0, 0.0)
    assert np.max(np.abs(out - sig)) < 1e-6
