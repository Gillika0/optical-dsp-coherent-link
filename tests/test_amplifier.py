"""EDFA and OSNR tests."""

from __future__ import annotations

import numpy as np
from optical_dsp.physics.amplifier import (
    ErbiumAmplifier,
    add_ase_to_osnr,
    ase_noise_power,
    bose_einstien_factor,
)
from optical_dsp.utils import db_to_linear, ref_bandwidth_hz


def test_bose_einstein_factor_positive() -> None:
    assert bose_einstien_factor(20.0, 5.0) > 0.0
    # high-gain asymptote: n_sp -> (nf_lin - 1)/2
    nf_lin = db_to_linear(6.0)
    assert np.isclose(bose_einstien_factor(60.0, 6.0), (nf_lin - 1.0) / 2.0, atol=0.02)


def test_ase_noise_power_scales_with_bandwidth() -> None:
    p1 = ase_noise_power(20.0, 5.0, 1550e-9, 12.5e9)
    p2 = ase_noise_power(20.0, 5.0, 1550e-9, 25.0e9)
    assert np.isclose(p2 / p1, 2.0, rtol=1e-9)


def test_osnr_mode_hits_target() -> None:
    sps, n_symbols = 4, 2048
    fs = 32e9 * sps
    rng = np.random.default_rng(1)
    sig = rng.normal(0.0, 0.1, n_symbols * sps) + 1j * rng.normal(0.0, 0.1, n_symbols * sps)
    sig = sig.astype(np.complex128)

    amp = ErbiumAmplifier(gain_db=16.0, target_osnr_db=20.0, seed=3)
    out_x, out_y = amp.amplify(sig, sig, fs)

    ref_bw = ref_bandwidth_hz(1550e-9, 0.1)
    p_total = float(np.mean(np.abs(out_x) ** 2 + np.abs(out_y) ** 2))
    noise = out_x - sig * np.sqrt(db_to_linear(16.0))
    n0 = float(np.mean(np.abs(noise) ** 2)) / fs  # one-sided PSD per pol [W/Hz]
    osnr_lin = p_total / (2.0 * n0 * ref_bw)
    measured = 10.0 * np.log10(osnr_lin)
    assert np.isclose(measured, 20.0, atol=0.5)


def test_add_ase_to_osnr_noise_power() -> None:
    fs = 32e9 * 4
    n = 4096
    rng = np.random.default_rng(2)
    sig = (rng.normal(0.0, 0.1, n) + 1j * rng.normal(0.0, 0.1, n)).astype(np.complex128)
    sig *= np.sqrt(0.5 / np.mean(np.abs(sig) ** 2))  # unit total power
    out_x, out_y = add_ase_to_osnr(sig, sig * 0, 20.0, fs, seed=5)
    noise = out_x - sig
    ref_bw = ref_bandwidth_hz(1550e-9, 0.1)
    p_total = float(np.mean(np.abs(sig) ** 2))  # signal power on pol X only
    n0 = float(np.mean(np.abs(noise) ** 2)) / fs
    osnr = 10.0 * np.log10(p_total / (2.0 * n0 * ref_bw))
    assert np.isclose(osnr, 20.0, atol=0.5)


def test_osnr_mode_splits_ase_over_amplifiers() -> None:
    sps, n_symbols = 4, 2048
    fs = 32e9 * sps
    rng = np.random.default_rng(1)
    sig = rng.normal(0.0, 0.1, n_symbols * sps) + 1j * rng.normal(0.0, 0.1, n_symbols * sps)
    sig = sig.astype(np.complex128)

    ref_bw = ref_bandwidth_hz(1550e-9, 0.1)
    for n_amp in (1, 4, 8):
        amp = ErbiumAmplifier(gain_db=16.0, target_osnr_db=20.0, n_amplifiers=n_amp, seed=3)
        out_x, out_y = amp.amplify(sig, sig, fs)
        # the per-stage ASE budget is n0/N, so a *single* stage only produces
        # a fraction of the noise: the OSNR of the first stage is higher
        p_total = float(np.mean(np.abs(out_x) ** 2 + np.abs(out_y) ** 2))
        noise = out_x - sig * np.sqrt(db_to_linear(16.0))
        n0_stage = float(np.mean(np.abs(noise) ** 2)) / fs
        osnr_stage = 10.0 * np.log10(p_total / (2.0 * n0_stage * ref_bw))
        assert np.isclose(osnr_stage, 20.0 + 10.0 * np.log10(n_amp), atol=0.5)
