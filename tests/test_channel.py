"""SSFM channel vs analytical linear reference tests."""

from __future__ import annotations

import numpy as np
from optical_dsp.physics.channel import (
    SsfmChannel,
    apply_analytical_linear_channel,
)
from optical_dsp.physics.transmitter import rrc_taps
from optical_dsp.utils import QPSK, FibreParams, dbm_to_w


def _qpsk_field(n_symbols: int, sps: int, power_dbm: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, n_symbols)]
    up = np.zeros(n_symbols * sps, dtype=np.complex128)
    up[::sps] = sym
    h = rrc_taps(0.2, sps, 33)
    conv = np.convolve(up, h)
    return np.sqrt(dbm_to_w(power_dbm)) * conv[(len(h) - 1) // 2 :][: len(up)]


def test_ssfm_matches_analytical_no_nonlinearity() -> None:
    sps, n_symbols = 4, 2048
    fibre = FibreParams(nonlinear_index_m2_w=0.0)
    sig = _qpsk_field(n_symbols, sps, 0.0)
    fs = 32e9 * sps

    ref = apply_analytical_linear_channel(sig, fs, 80.0, fibre)
    sim = SsfmChannel(fibre, dz_km=0.5, manakov=True).propagate(sig, sig, 80.0, fs)[0]

    err = np.linalg.norm(sim - ref) / np.linalg.norm(ref)
    assert err < 1e-6


def test_ssfm_attenuation() -> None:
    sps, n_symbols = 4, 1024
    fibre = FibreParams(alpha_db_km=0.2, nonlinear_index_m2_w=0.0)
    sig = _qpsk_field(n_symbols, sps, 3.0)
    fs = 32e9 * sps
    out = SsfmChannel(fibre, dz_km=0.5, manakov=True).propagate(sig, sig, 80.0, fs)[0]
    loss_db = 10.0 * np.log10(np.mean(np.abs(sig) ** 2) / np.mean(np.abs(out) ** 2))
    assert np.isclose(loss_db, 0.2 * 80.0, atol=0.05)


def test_analytical_transfer_unit_dispersion_zero() -> None:
    fibre = FibreParams(dispersion_ps=0.0, alpha_db_km=0.0, nonlinear_index_m2_w=0.0)
    sig = _qpsk_field(512, 4, 0.0)
    out = apply_analytical_linear_channel(sig, 32e9 * 4, 40.0, fibre)
    assert np.allclose(out, sig, atol=1e-12)
