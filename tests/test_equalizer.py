"""MIMO equalizer tests: AGC, latency alignment, CMA/DD steady states."""

from __future__ import annotations

import numpy as np
from optical_dsp.analysis.metrics import evm_rms, resolve_rotation
from optical_dsp.dsp.equalizer import MimoEqualizer
from optical_dsp.utils import QAM16, QPSK

LAT = (15 - 1) // 2  # the equalizer's intrinsic centre-tap latency


def _probe_evm(y0, constellation, reference):
    """Rotation-resolved EVM on the latency-aligned trailing window."""
    a = y0[LAT:]
    ref = reference[: len(a)]
    tail = slice(-3000, None)
    k = resolve_rotation(a[tail], constellation, ref[tail])
    return evm_rms(a[tail] * np.exp(1j * np.pi * k / constellation.symmetry_order), ref[tail])


def test_agc_gain_value_matches_power() -> None:
    rng = np.random.default_rng(1)
    n = 4096
    x0 = (rng.normal(0.0, 0.1, n) + 1j * rng.normal(0.0, 0.1, n)).astype(np.complex128)
    x1 = 0.5 * x0
    eq = MimoEqualizer(n_taps=7, seed=1)
    eq._stacked_inputs(x0, x1)
    power = float(np.mean(np.abs(x0) ** 2 + np.abs(x1) ** 2))
    assert np.isclose(eq.input_gain, 1.0 / np.sqrt(power))


def test_agc_can_be_disabled() -> None:
    x0 = np.ones(64, dtype=np.complex128)
    x1 = np.ones(64, dtype=np.complex128)
    eq = MimoEqualizer(n_taps=7, seed=2, normalise_input=False)
    eq._stacked_inputs(x0, x1)
    assert eq.input_gain == 1.0


def test_cma_qpsk_preserves_clean_signal() -> None:
    rng = np.random.default_rng(0)
    const = QPSK()
    n = 16384
    s0 = const.symbols[rng.integers(0, 4, n)]
    s1 = const.symbols[rng.integers(0, 4, n)]
    eq = MimoEqualizer(n_taps=15, mu_cma=1e-3, seed=1)
    y0, y1, _ = eq.equalize(s0, s1, const, n_cma=None)
    assert _probe_evm(y0, const, s0) < 0.5
    assert _probe_evm(y1, const, s1) < 0.5


def test_cma_qpsk_meets_noise_floor() -> None:
    rng = np.random.default_rng(0)
    const = QPSK()
    n = 16384
    s0 = const.symbols[rng.integers(0, 4, n)]
    s1 = const.symbols[rng.integers(0, 4, n)]
    noise = (rng.normal(0.0, 0.2, n) + 1j * rng.normal(0.0, 0.2, n)).astype(np.complex128)
    eq = MimoEqualizer(n_taps=15, mu_cma=1e-3, seed=1)
    y0, _, _ = eq.equalize(s0 + noise, s1, const, n_cma=None)
    # input EVM ~28%, equalized EVM must stay at the same noise level
    assert _probe_evm(y0, const, s0) < 35.0


def test_qam16_blind_preserves_clean_signal() -> None:
    rng = np.random.default_rng(0)
    const = QAM16()
    n = 16384
    s0 = const.symbols[rng.integers(0, 16, n)]
    s1 = const.symbols[rng.integers(0, 16, n)]
    eq = MimoEqualizer(n_taps=15, seed=1)
    y0, y1, _ = eq.equalize(s0, s1, const, n_cma=1200)
    # MMA's single modulus cannot be exactly zero on the two-ring 16-QAM
    # lattice (its fixed point sits ~4.5% off the ideal points); the blind
    # chain must preserve the signal within that intrinsic residual.
    assert _probe_evm(y0, const, s0) < 8.0
    assert _probe_evm(y1, const, s1) < 8.0


def test_qam16_blind_meets_noise_floor() -> None:
    rng = np.random.default_rng(0)
    const = QAM16()
    n = 16384
    s0 = const.symbols[rng.integers(0, 16, n)]
    s1 = const.symbols[rng.integers(0, 16, n)]
    noise = (rng.normal(0.0, 0.12, n) + 1j * rng.normal(0.0, 0.12, n)).astype(np.complex128)
    eq = MimoEqualizer(n_taps=15, seed=1)
    y0, _, _ = eq.equalize(s0 + noise, s1, const, n_cma=1200)
    assert _probe_evm(y0, const, s0) < 25.0


def test_equalizer_outputs_are_intrinsically_latency_shifted() -> None:
    # Sanity: the first LAT samples correspond to the zero-history padding.
    rng = np.random.default_rng(0)
    const = QPSK()
    n = 2048
    s0 = const.symbols[rng.integers(0, 4, n)]
    s1 = const.symbols[rng.integers(0, 4, n)]
    eq = MimoEqualizer(n_taps=15, seed=1)
    y0, _, _ = eq.equalize(s0, s1, const, n_cma=None)
    # with zero adaptation the filter acts as pure identity: aligned block matches
    corr = abs(np.vdot(y0[LAT:], s0[: len(y0) - LAT])) / np.sqrt(
        np.vdot(y0[LAT:], y0[LAT:]).real * np.vdot(s0[: len(y0) - LAT], s0[: len(y0) - LAT]).real
    )
    assert corr > 0.99
