"""Constellation and unit-conversion tests."""

from __future__ import annotations

import numpy as np
import pytest
from optical_dsp.utils import (
    QAM16,
    QAM64,
    QPSK,
    Constellation,
    beta2_from_D,
    db_to_linear,
    dbm_to_w,
    get_constellation,
    linear_to_db,
    w_to_dbm,
)


@pytest.mark.parametrize("factory", [QPSK, QAM16, QAM64])
def test_unit_energy(factory) -> None:
    const = factory()
    assert np.isclose(np.mean(np.abs(const.symbols) ** 2), 1.0)


@pytest.mark.parametrize("factory", [QPSK, QAM16, QAM64])
def test_symmetry_is_quarter_turn(factory) -> None:
    const = factory()
    assert const.symmetry_order == 4
    # quarter-turn invariance of the constellation set
    assert np.allclose(
        np.sort(np.angle(const.symbols * np.exp(1j * np.pi / 2))), np.sort(np.angle(const.symbols))
    )


def test_bits_roundtrip_qam16() -> None:
    rng = np.random.default_rng(0)
    const = QAM16()
    bits = rng.integers(0, 2, 4096, dtype=np.uint8)
    sym = const.bits_to_symbols(bits)
    idx = const.nearest_index(sym)
    assert np.array_equal(const.symbols_to_bits(idx), bits)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_decision_radius(order: int) -> None:
    const = Constellation(order=order)
    idx = np.arange(order, dtype=np.int64)
    sym = const.symbols[idx]
    # correct symbols are the closest to themselves
    assert np.array_equal(const.nearest_index(sym), idx)


def test_get_constellation_names() -> None:
    assert get_constellation("QPSK") is QPSK()
    assert get_constellation("16-qam") is QAM16()
    assert get_constellation("64QAM") is QAM64()
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
