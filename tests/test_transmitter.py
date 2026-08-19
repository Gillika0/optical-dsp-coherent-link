"""Transmitter / RRC / helper tests."""

from __future__ import annotations

import numpy as np
from optical_dsp.physics.transmitter import CoherentTransmitter, generate_prbs, rrc_taps
from optical_dsp.utils import QAM16, QPSK, dbm_to_w


def test_prbs_reproducible() -> None:
    a = generate_prbs(2000, seed=1)
    b = generate_prbs(2000, seed=1)
    c = generate_prbs(2000, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_prbs_balance() -> None:
    bits = generate_prbs(1 << 14, seed=7).astype(np.float64)
    assert np.abs(bits.mean() - 0.5) < 0.02
    assert np.any(bits == 1) and np.any(bits == 0)


def test_rrc_energy_normalisation() -> None:
    sps = 4
    h = rrc_taps(beta=0.2, sps=sps, n_taps=33)
    assert h.shape == (33,)
    assert np.isclose(np.sum(h**2), float(sps))  # unit energy per symbol
    assert np.isclose(h.max(), np.abs(h).max())


def test_rrc_zero_rolloff_is_sinc() -> None:
    sps, n_taps = 4, 33
    h = rrc_taps(beta=0.0, sps=sps, n_taps=n_taps)
    x = (np.arange(n_taps) - (n_taps - 1) / 2) / sps
    reference = np.ones_like(x)
    nz = ~np.isclose(x, 0.0)
    reference[nz] = np.sin(np.pi * x[nz]) / (np.pi * x[nz])
    reference *= np.sqrt(sps / np.sum(reference**2))
    assert np.allclose(h, reference, atol=1e-3)


def test_pdm_power_balance_and_budget() -> None:
    tx = CoherentTransmitter(QPSK(), symbol_rate=32e9, power_dbm=3.0, seed=1)
    t = tx.transmit(16384, seed=1)
    p_x = float(np.mean(np.abs(t.ex) ** 2))
    p_y = float(np.mean(np.abs(t.ey) ** 2))
    assert np.isclose(p_x, p_y, rtol=1e-3)
    # total launch power exactly matches the dBm setting
    assert np.isclose(p_x + p_y, dbm_to_w(3.0), rtol=0.01)


def test_transmit_references() -> None:
    tx = CoherentTransmitter(QAM16(), symbol_rate=32e9, seed=5)
    t = tx.transmit(2048, seed=5)
    assert t.sx.shape == (2048,)
    assert t.bx.shape == (2048 * 4,)
    assert len(t.ex) == 2048 * tx.sps
    assert t.fs == 32e9 * tx.sps


def test_rrc_beta_zero_allowed() -> None:
    h = rrc_taps(beta=0.0, sps=4, n_taps=7)
    assert np.all(np.isfinite(h))
