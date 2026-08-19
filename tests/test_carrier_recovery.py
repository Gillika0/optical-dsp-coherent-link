"""Carrier-recovery tests: FOE accuracy and BPS phase removal."""

from __future__ import annotations

import numpy as np
from optical_dsp.analysis.metrics import evm_rms, resolve_rotation
from optical_dsp.dsp.carrier_recovery import (
    BlindPhaseSearch,
    FrequencyOffsetEstimator,
    estimate_frequency_offset,
    remove_frequency_offset,
)
from optical_dsp.dsp.front_end import matched_filter_and_retime
from optical_dsp.physics.transmitter import phase_noise_common
from optical_dsp.utils import QAM16, QPSK


def _shaped_qpsk(n_sym, sps=4, beta=0.2):
    from optical_dsp.physics.transmitter import rrc_taps

    rng = np.random.default_rng(0)
    sym = QPSK().symbols[rng.integers(0, 4, n_sym)]
    h = rrc_taps(beta=beta, sps=sps, n_taps=33)
    up = np.zeros(n_sym * sps, dtype=np.complex128)
    up[::sps] = sym
    return np.convolve(up, h, mode="full")[: n_sym * sps], sym


def test_foe_estimates_tone() -> None:
    fs = 128e9
    n = 2**16
    f0 = 0.2e9
    t = np.arange(n) / fs
    sig = np.exp(1j * 2 * np.pi * f0 * t) + 0.01 * np.exp(1j * 2 * np.pi * 0.31e9 * t)
    est = estimate_frequency_offset(sig, fs)
    assert np.isclose(est, f0, atol=fs / n)


def test_foe_on_qpsk_signal() -> None:
    fs = 128e9
    sig, _ = _shaped_qpsk(2**13)
    n = len(sig)
    t = np.arange(n) / fs
    sig = sig * np.exp(1j * 2 * np.pi * 0.25e9 * t)
    est = estimate_frequency_offset(sig, fs)
    assert np.isclose(est / 1e9, 0.25, atol=0.01)


def test_foe_compensator_removes_rotation() -> None:
    fs = 128e9
    sig, sym = _shaped_qpsk(2**13)
    n = len(sig)
    t = np.arange(n) / fs
    sig = sig * np.exp(1j * 2 * np.pi * 0.15e9 * t)

    foe = FrequencyOffsetEstimator()
    f_est = foe.estimate(sig, None, fs)
    corr, _ = foe.compensate(sig, np.zeros_like(sig), f_est, fs)
    resampled, _ = matched_filter_and_retime(corr, 4, beta=0.2, selection="peak")
    # residual phase per symbol after de-rotation must stay flat (~0 Hz)
    ph = np.unwrap(np.angle(resampled * np.conj(sym)))
    residual_slope = np.polyfit(np.arange(len(ph)), ph, 1)[0]
    assert abs(residual_slope) < 0.01  # rad/sample -> ~0 Hz


def test_bps_removes_static_rotation_qpsk() -> None:
    rng = np.random.default_rng(0)
    n = 8192
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, n)]
    rot = np.exp(1j * 1.13)  # arbitrary static phase
    rx = sym * rot
    z = BlindPhaseSearch(const, n_phases=64, block_size=32).run(rx)
    k = resolve_rotation(z, const, sym)
    assert evm_rms(z * np.exp(1j * 2 * np.pi * k / 4), sym) < 4.0


def test_bps_removes_static_rotation_qam16() -> None:
    rng = np.random.default_rng(1)
    n = 8192
    const = QAM16()
    sym = const.symbols[rng.integers(0, 16, n)]
    rx = sym * np.exp(1j * 1.2)
    z = BlindPhaseSearch(const, n_phases=64, block_size=16).run(rx)
    k = resolve_rotation(z, const, sym)
    assert evm_rms(z * np.exp(1j * 2 * np.pi * k / 4), sym) < 4.0


def test_bps_tracks_slow_wienner_drift() -> None:
    rng = np.random.default_rng(2)
    n = 16384
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, n)]
    phi = phase_noise_common(n, 32e9, linewidth_hz=300e3, seed=5)
    rx = sym * np.exp(1j * phi)
    z = BlindPhaseSearch(const, n_phases=64, block_size=32).run(rx)
    k = resolve_rotation(z, const, sym)
    evm = evm_rms(z * np.exp(1j * 2 * np.pi * k / 4), sym)
    assert evm < 5.0


def test_remove_frequency_offset_matches_fft_peak() -> None:
    fs = 128e9
    n = 2**14
    f0 = 0.1e9
    t = np.arange(n) / fs
    sig = np.exp(1j * 2 * np.pi * f0 * t)
    clean = remove_frequency_offset(sig, f0, fs)
    assert np.allclose(clean, np.ones(n), atol=1e-9)
