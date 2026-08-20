"""Constellation and unit-conversion tests."""

from __future__ import annotations

import numpy as np
import pytest
from optical_dsp.utils import (
    QAM8,
    QAM16,
    QAM64,
    QPSK,
    beta2_from_D,
    db_to_linear,
    dbm_to_w,
    get_constellation,
    linear_to_db,
    osnr_db_to_snr_db,
    w_to_dbm,
)


@pytest.mark.parametrize("factory", [QPSK, QAM8, QAM16, QAM64])
def test_unit_energy(factory) -> None:
    const = factory()
    assert np.isclose(np.mean(np.abs(const.symbols) ** 2), 1.0)


def test_8qam_geometry() -> None:
    const = QAM8()
    assert const.order == 8
    assert const.dims == (2, 4)
    assert const.symmetry_order == 1  # the 2x4 cross is not 90-degree invariant
    # unit-power scale: I in {+-1/sqrt(6)}, Q in {+-1/sqrt(6), +-3/sqrt(6)}
    assert np.allclose(np.unique(const.symbols.real), [-1.0, 1.0] / np.sqrt(6.0))
    assert np.allclose(np.unique(const.symbols.imag), [-3.0, -1.0, 1.0, 3.0] / np.sqrt(6.0))


@pytest.mark.parametrize("factory", [QPSK, QAM8, QAM16, QAM64])
def test_bits_roundtrip(factory) -> None:
    rng = np.random.default_rng(0)
    const = factory()
    bits = rng.integers(0, 2, const.order * 256 * const.bits_per_symbol, dtype=np.uint8)
    sym = const.bits_to_symbols(bits)
    idx = const.nearest_index(sym)
    assert np.array_equal(const.symbols_to_bits(idx), bits)


@pytest.mark.parametrize("factory", [QPSK, QAM8, QAM16, QAM64])
def test_decision_radius(factory) -> None:
    const = factory()
    idx = np.arange(const.order, dtype=np.int64)
    sym = const.symbols[idx]
    # correct symbols are the closest to themselves
    assert np.array_equal(const.nearest_index(sym), idx)


def test_get_constellation_names() -> None:
    assert get_constellation("QPSK") is QPSK()
    assert get_constellation("16-qam") is QAM16()
    assert get_constellation("64QAM") is QAM64()
    assert get_constellation("8-QAM") is QAM8()
    assert get_constellation("DP-QPSK") is QPSK()
    assert get_constellation("DP-8QAM") is QAM8()
    assert get_constellation("DP-16QAM") is QAM16()
    assert get_constellation("dp-64qam") is QAM64()
    with pytest.raises(KeyError):
        get_constellation("8PSK")


def test_power_conversions() -> None:
    assert np.isclose(dbm_to_w(0.0), 1e-3)
    assert np.isclose(w_to_dbm(1e-3), 0.0)
    assert np.isclose(dbm_to_w(w_to_dbm(2.3)), 2.3)


def test_db_conversions() -> None:
    assert np.isclose(db_to_linear(10.0), 10.0)
    assert np.isclose(linear_to_db(100.0), 20.0)


def test_beta2_sign_and_value() -> None:
    # SMF-28: D=16.6 ps/nm/km at 1550 nm -> beta2 ~ -21.3e-27 s^2/m
    b2 = beta2_from_D(1550e-9, 16.6)
    assert b2 < 0.0
    assert np.isclose(b2, -21.3e-27, atol=0.4e-27)


def test_mma_radii_qam16() -> None:
    const = QAM16()
    assert const.decision_radius2_i == pytest.approx(const.decision_radius2_q)
    assert const.cma_radius2 > const.decision_radius2_i


def test_osnr_to_snr_matches_amplifier_convention() -> None:
    # OSNR 20 dB, 32 GBd, 0.1 nm (12.5 GHz) ref. bandwidth:
    # Es/N0 = OSNR + 10*log10(12.5/32) = 20 - 4.08 = 15.92 dB.
    snr = osnr_db_to_snr_db(20.0, 32e9, 12.5e9)
    assert np.isclose(snr, 20.0 + 10.0 * np.log10(12.5e9 / 32e9), atol=1e-9)
    # a dual-pol-noise OSNR convention costs exactly 10*log10(npol)
    assert np.isclose(
        osnr_db_to_snr_db(20.0, 32e9, 12.5e9, npol=2), snr - 10.0 * np.log10(2.0), atol=1e-9
    )
