"""Metrics tests: EVM, Q-factor, BER, rotation resolution."""

from __future__ import annotations

import numpy as np
from optical_dsp.analysis.metrics import (
    evm_rms,
    measure_ber,
    q_factor_from_ber,
    q_function,
    resolve_rotation,
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
