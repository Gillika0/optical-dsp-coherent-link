"""Metrics tests: EVM, Q-factor, BER, rotation resolution."""

from __future__ import annotations

import numpy as np
import pytest
from optical_dsp.analysis.metrics import (
    HD_FEC_RS255_239,
    STRONG_FEC_RS255_213,
    adc_target_metrics,
    apply_fec,
    evm_rms,
    fec_coding_gain_db,
    line_rate_gbps,
    measure_ber,
    net_rate_gbps,
    q_factor_from_ber,
    q_function,
    required_osnr_db,
    resolve_fec,
    resolve_rotation,
    spectral_efficiency_bits_s_hz,
    theoretical_ber_from_evm,
    theoretical_ber_qam,
)
from optical_dsp.utils import QPSK


def test_q_function_values() -> None:
    assert np.isclose(q_function(0.0), 0.5)
    assert np.isclose(q_function(3.0), 0.00135, atol=1e-4)
    assert q_function(3.0) > 0.0


def test_evm_rms_clean() -> None:
    rng = np.random.default_rng(0)
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, 1024)]
    assert evm_rms(sym, sym) < 1e-12
    assert np.isclose(evm_rms(2.0 * sym, sym, scale_optimum=True), 0.0, atol=1e-9)


def test_evm_rms_noise() -> None:
    rng = np.random.default_rng(1)
    n = 20000
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, n)]
    noise = (rng.normal(0.0, 0.05, n) + 1j * rng.normal(0.0, 0.05, n)).astype(np.complex128)
    evm = evm_rms(sym + noise, sym)
    # E|n|^2 = 2*0.05^2 = 0.005 -> EVM ~ sqrt(0.005) = 7.1%
    assert np.isclose(evm, 100.0 * np.sqrt(0.005), atol=0.3)


def test_resolve_rotation_finds_quadrant() -> None:
    rng = np.random.default_rng(0)
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, 2048)]
    rot = sym * np.exp(1j * 2 * np.pi * 2 / 4)  # 180 deg
    k = resolve_rotation(rot, const, sym)
    assert k == 2
    k0 = resolve_rotation(sym, const, sym)
    assert k0 == 0


def test_measure_ber_clean_and_ambiguous() -> None:
    rng = np.random.default_rng(0)
    const = QPSK()
    n = 4096
    bits = rng.integers(0, 2, n * 2, dtype=np.uint8)
    sym = const.bits_to_symbols(bits)

    res = measure_ber(sym, const, bits, resolve_rotation=True)
    assert res.ber == 0.0 and res.n_errors == 0

    # rotated by 90 deg: BER would be catastrophic without resolution
    res_rot = measure_ber(sym * np.exp(1j * np.pi / 2), const, bits, resolve_rotation=True)
    assert res_rot.ber == 0.0
    bad = measure_ber(sym * np.exp(1j * np.pi / 2), const, bits, resolve_rotation=False)
    assert bad.ber > 0.4


def test_q_factor_monotonic() -> None:
    assert q_factor_from_ber(1e-6) > q_factor_from_ber(1e-3)
    assert q_factor_from_ber(1e-9) > 0.0


def test_theoretical_ber_sanity() -> None:
    assert theoretical_ber_qam(30.0, 4) < theoretical_ber_qam(10.0, 4)
    assert 0.0 < theoretical_ber_qam(15.0, 16) < 0.5
    assert theoretical_ber_from_evm(1.0) < 1e-12


@pytest.mark.parametrize("order", [4, 8, 16, 64])
def test_theoretical_ber_monotonic_each_order(order: int) -> None:
    assert theoretical_ber_qam(28.0, order) < theoretical_ber_qam(10.0, order)
    assert 0.0 <= theoretical_ber_qam(0.0, order) <= 0.5


def test_theoretical_ber_8qam_reference_value() -> None:
    # rectangular 2x4 cross, Q(sqrt(Es/N0/3)) tail: at 15 dB -> ~4.86e-4
    assert np.isclose(theoretical_ber_qam(15.0, 8), 4.86e-4, rtol=0.05)


def test_kpi_helpers() -> None:
    # dual-pol 32 GBd: QPSK 128 Gb/s, 16-QAM 256 Gb/s; SE 4 / 8 bit/s/Hz
    assert np.isclose(line_rate_gbps(32e9, 4), 128.0)
    assert np.isclose(line_rate_gbps(32e9, 16), 256.0)
    assert np.isclose(spectral_efficiency_bits_s_hz(4, npol=2), 4.0)
    assert np.isclose(spectral_efficiency_bits_s_hz(16, npol=2), 8.0)
    # HD-FEC 7%: net = raw * 239/255
    assert np.isclose(net_rate_gbps(128.0, HD_FEC_RS255_239), 128.0 * 239 / 255)
    assert np.isclose(net_rate_gbps(128.0, None), 128.0)


def test_required_osnr_reference() -> None:
    # QPSK 32 GBd, dual pol, HD-FEC threshold (3.8e-3) in 0.1 nm.
    # QPSK needs Es/N0 ~ 8.5 dB at 3.8e-3; OSNR = SNR + 10log10(Rs*npol/B_ref).
    osnr = required_osnr_db(32e9, 4, 3.8e-3, 12.5e9, npol=2)
    expected = 8.50 + 10.0 * np.log10(32e9 * 2.0 / 12.5e9)
    assert np.isclose(osnr, expected, atol=0.1)


def test_fec_coding_gain_range() -> None:
    hd = fec_coding_gain_db(4, HD_FEC_RS255_239)
    strong = fec_coding_gain_db(4, STRONG_FEC_RS255_213)
    assert 5.0 < hd < 7.0  # textbook HD-FEC gain ~5.8 dB at 1e-12
    assert strong > hd + 1.0  # deeper code buys more gain


def test_resolve_fec() -> None:
    assert resolve_fec("none") is None
    assert resolve_fec("hd") is HD_FEC_RS255_239
    assert resolve_fec("strong") is STRONG_FEC_RS255_213


def test_fec_codes_have_expected_capability() -> None:
    assert HD_FEC_RS255_239.t == 8
    assert np.isclose(HD_FEC_RS255_239.overhead, 16.0 / 239.0)
    assert STRONG_FEC_RS255_213.t == 21
    assert np.isclose(STRONG_FEC_RS255_213.overhead, 42.0 / 213.0, atol=1e-9)


def test_post_fec_ber_drops_in_correctable_range() -> None:
    # well inside the code capability: BER collapses toward the display floor
    assert apply_fec(1e-4, HD_FEC_RS255_239) < 1e-10
    assert apply_fec(1e-3, STRONG_FEC_RS255_213) < 1e-10
    # the strong code handles an input the 7% code cannot
    assert apply_fec(5e-3, STRONG_FEC_RS255_213) < apply_fec(5e-3, HD_FEC_RS255_239)


def test_post_fec_ber_monotonic_and_bounded() -> None:
    prev = -1.0
    for p in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.4):
        post = apply_fec(p, HD_FEC_RS255_239)
        assert 0.0 <= post <= 0.5
        assert post >= prev
        prev = post
    assert apply_fec(0.0, HD_FEC_RS255_239) == 0.0
    assert apply_fec(0.5, HD_FEC_RS255_239) == 0.5


def test_adc_target_metrics_symmetric_clean_levels() -> None:
    # ideal QPSK at 4 samples/symbol: symmetric rails, no clipping
    rng = np.random.default_rng(0)
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, 8192)]
    sig = np.repeat(sym, 4)
    sig = sig + 0.001 * (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size))
    i_res, q_res = adc_target_metrics(sig, 4)
    assert abs(i_res.level_pos + i_res.level_neg) < 1e-3  # symmetric around 0
    assert np.isclose(i_res.spacing, i_res.level_pos - i_res.level_neg)
    assert np.isclose(q_res.spacing, i_res.spacing, rtol=0.05)
    assert i_res.clip_fraction == 0.0
    assert i_res.eye_opening_db > 10.0


def test_adc_target_metrics_tiny_signal_returns_nan() -> None:
    i_res, q_res = adc_target_metrics(np.zeros(8, dtype=np.complex128), 4)
    assert np.isnan(i_res.vpp)
    assert np.isnan(q_res.vpp)
