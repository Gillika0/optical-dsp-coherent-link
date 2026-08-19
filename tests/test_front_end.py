"""Coherent front-end tests: IQ imbalance, ADC, LO offset, matched filter."""

from __future__ import annotations

import numpy as np
from optical_dsp.dsp.carrier_recovery import estimate_frequency_offset
from optical_dsp.dsp.front_end import (
    CoherentFrontend,
    adc_quantize,
    apply_iq_imbalance,
    matched_filter_and_retime,
)
from optical_dsp.physics.laser import Laser
from optical_dsp.physics.transmitter import CoherentTransmitter, rrc_taps
from optical_dsp.utils import QPSK


def test_iq_imbalance_identity() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 1.0, 256) + 1j * rng.normal(0.0, 1.0, 256)
    assert np.allclose(apply_iq_imbalance(s, 0.0, 0.0), s)


def test_iq_imbalance_distorts() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 1.0, 256) + 1j * rng.normal(0.0, 1.0, 256)
    out = apply_iq_imbalance(s, 3.0, 5.0)
    assert np.mean(np.abs(out - s) ** 2) > 1e-6


def test_adc_quantization_levels() -> None:
    rng = np.random.default_rng(0)
    s = (rng.normal(0.0, 1.0, 4096) + 1j * rng.normal(0.0, 1.0, 4096)).astype(np.complex128)
    q = adc_quantize(s, bits=8)
    step = 3.5 / 128.0  # one LSB with default 3.5-sigma full-scale
    # nearly all samples within half an LSB; outliers only from clipping
    assert np.percentile(np.abs(q - s), 99.9) < step
    assert np.max(np.abs(q - s)) < 1.0


def test_lo_frequency_offset_shows_on_photocurrent() -> None:
    tx = CoherentTransmitter(QPSK(), symbol_rate=32e9, sps=4, power_dbm=0.0, seed=1)
    t = tx.transmit(1024, seed=1)
    fs = t.fs
    fe = CoherentFrontend(lo=Laser(power_dbm=10.0, linewidth_khz=0.0, seed=1), adc_bits=None)
    r_x, _ = fe.detect(t.ex, t.ey, fs)
    fe_off = CoherentFrontend(
        lo=Laser(power_dbm=10.0, linewidth_khz=0.0, seed=1),
        adc_bits=None,
        lo_freq_offset_hz=0.2e9,
    )
    r_xo, _ = fe_off.detect(t.ex, t.ey, fs)
    est = estimate_frequency_offset(r_xo, fs)
    assert np.isclose(est / 1e9, 0.2, atol=0.02)
    # without offset the estimate should be ~0
    assert abs(estimate_frequency_offset(r_x, fs)) < 1e8


def test_matched_filter_retimes() -> None:
    n = 512
    sps = 4
    rng = np.random.default_rng(0)
    sym = QPSK().symbols[rng.integers(0, 4, n)]
    h = rrc_taps(beta=0.2, sps=sps, n_taps=33)
    up = np.zeros(n * sps, dtype=np.complex128)
    up[::sps] = sym
    filtered = np.convolve(up, h, mode="full")[: n * sps]
    delayed = np.roll(filtered, 1)
    out, offset = matched_filter_and_retime(delayed, sps, beta=0.2, selection="peak")
    assert out.shape == (n,)
    assert 0 <= offset < sps
    # retiming must land on the RRC peaks: constant modulus, near-constant amplitude
    mid = out[8:]  # skip RRC edge transients
    gain = np.mean(np.abs(mid))
    assert gain > 2.0  # matched response ~ sum(h^2) = sps at the peak grid
    assert np.std(np.abs(mid)) / gain < 0.05
