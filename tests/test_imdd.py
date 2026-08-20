"""IM/DD (short-reach / PON) engine tests."""

from __future__ import annotations

import numpy as np
from optical_dsp.imdd import (
    ImddConfig,
    analytical_sensitivity_dbm,
    imdd_sensitivity,
    link_budget,
    received_power_dbm,
    run_imdd,
)

_FAST = dict(n_symbols=2**12, sps=8, seed=7)


def _cfg(**overrides: object) -> ImddConfig:
    return ImddConfig(**_FAST, **overrides)


def test_default_config_is_realistic_oband_pam4() -> None:
    cfg = ImddConfig()
    assert cfg.modulation == "PAM4"
    assert np.isclose(cfg.symbol_rate, 26.5625e9)
    assert cfg.laser_type == "EML"
    assert np.isclose(cfg.wavelength_nm, 1310.0)
    assert cfg.dispersion_ps_per_nm_km == 0.0  # O-band, D ~ 0
    assert cfg.length_km == 20.0


def test_imdd_back_to_back_eml_is_clean() -> None:
    res = run_imdd(_cfg(modulation="PAM4", laser_type="EML", length_km=0.0, receiver_type="PIN"))
    assert res.ber < 1e-3
    assert res.eye_opening["min_opening"] > 0.05
    assert res.eye_opening["eop_db"] < 3.0
    assert res.eye_opening_eq["eop_db"] < 3.0


def test_imdd_dml_chirp_penalty() -> None:
    cfg = _cfg(
        modulation="PAM4",
        length_km=10.0,
        wavelength_nm=1550.0,
        dispersion_ps_per_nm_km=17.0,
        receiver_type="PIN",
    )
    dml = run_imdd(ImddConfig(**{**cfg.__dict__, "laser_type": "DML", "chirp_alpha": 2.0}))
    eml = run_imdd(ImddConfig(**{**cfg.__dict__, "laser_type": "EML", "chirp_alpha": 0.0}))
    assert dml.ber >= eml.ber


def test_imdd_dispersion_fades_with_distance() -> None:
    b2b = run_imdd(
        _cfg(
            modulation="PAM4",
            laser_type="EML",
            length_km=0.0,
            wavelength_nm=1550.0,
            dispersion_ps_per_nm_km=17.0,
            receiver_type="PIN",
        )
    )
    far = run_imdd(
        _cfg(
            modulation="PAM4",
            laser_type="EML",
            length_km=20.0,
            wavelength_nm=1550.0,
            dispersion_ps_per_nm_km=17.0,
            receiver_type="PIN",
        )
    )
    assert far.ber >= b2b.ber


def test_imdd_equalizer_runs() -> None:
    for eq in ("None", "FFE", "DFE"):
        res = run_imdd(_cfg(modulation="PAM4", laser_type="EML", length_km=10.0, equalizer_type=eq))
        assert 0.0 <= res.ber <= 0.5


def test_imdd_eye_metrics_shape() -> None:
    res = run_imdd(_cfg(modulation="PAM4", laser_type="EML", length_km=10.0))
    eye = res.eye_opening
    assert len(eye["thresholds"]) == 3
    assert len(eye["levels"]) == 4
    assert len(eye["openings"]) == 3
    assert eye["eop_db"] >= 0.0
    assert 0.0 <= eye["eye_linearity"] <= 1.0
    eq = res.eye_opening_eq
    assert len(eq["thresholds"]) == 3
    assert len(eq["levels"]) == 4
    assert len(res.eye_eq) % res.config.sps == 0


def test_imdd_eop_clamped_at_10db() -> None:
    res = run_imdd(
        _cfg(
            modulation="PAM4",
            laser_type="EML",
            length_km=40.0,
            wavelength_nm=1550.0,
            dispersion_ps_per_nm_km=17.0,
        )
    )
    assert res.eye_opening["eop_db"] == 10.0
    assert res.eye_opening["eop_clamped"] is True
    assert res.eye_opening_eq["eop_db"] <= 10.0


def test_received_power_and_budget() -> None:
    cfg = _cfg(
        tx_power_dbm=3.0,
        connector_loss_db=1.0,
        alpha_db_km=0.2,
        length_km=10.0,
        splitter_ratio=16,
    )
    # 3 dBm - (1 dB connector + 2 dB fibre + 10*log10(16) splitter)
    splitter = 10.0 * np.log10(16)
    assert np.isclose(received_power_dbm(cfg), 3.0 - (1.0 + 2.0 + splitter), atol=1e-9)
    budget = link_budget(cfg)
    labels = [b[0] for b in budget]
    assert labels[-3] == "Received power (ROP)"
    assert labels[-2] == "Receiver sensitivity"
    assert labels[-1] == "System margin"
    assert np.isclose(budget[4][2], received_power_dbm(cfg), atol=1e-9)
    assert np.isclose(budget[5][2], analytical_sensitivity_dbm(cfg, 1e-3), atol=1e-6)
    # margin bar: relative increment ROP - sensitivity, cumulative at the ROP
    margin = budget[6]
    assert np.isclose(margin[1], budget[4][2] - budget[5][2], atol=1e-6)
    assert np.isclose(margin[2], budget[4][2], atol=1e-9)


def test_analytical_sensitivity_plausible() -> None:
    sens_pin = analytical_sensitivity_dbm(_cfg(receiver_type="PIN"))
    sens_apd = analytical_sensitivity_dbm(_cfg(receiver_type="APD"))
    assert -40.0 < sens_pin < 0.0
    assert sens_apd < sens_pin  # APD is more sensitive in the thermal-noise regime


def test_sensitivity_sweep_monotonic_and_crossing() -> None:
    cfg = _cfg(modulation="PAM4", laser_type="EML")
    rop, curves, crossings = imdd_sensitivity(
        cfg, rop_grid_dbm=[-30, -25, -20, -15, -10, -5], receivers=("PIN", "APD")
    )
    for rx in ("PIN", "APD"):
        ber = curves[rx]
        assert all(b2 <= b1 + 1e-9 for b1, b2 in zip(ber, ber[1:]))  # non-increasing
        assert crossings[rx] <= rop[-1]
    assert crossings["APD"] < crossings["PIN"]  # APD crosses the target earlier


def test_sensitivity_drops_below_1e6_without_floor() -> None:
    cfg = _cfg(modulation="PAM4", laser_type="EML", receiver_type="PIN")
    _, curves, _ = imdd_sensitivity(cfg, rop_grid_dbm=[-15, -10, -5, 0], receivers=("PIN", "APD"))
    for rx in ("PIN", "APD"):
        # high received power must beat the FEC threshold by far and keep falling
        assert curves[rx][-1] < 1e-6
        assert curves[rx][-1] < curves[rx][-2]


def test_imdd_modulation_levels() -> None:
    for mod in ("NRZ", "PAM4", "PAM8"):
        res = run_imdd(_cfg(modulation=mod, laser_type="EML", length_km=0.0, receiver_type="PIN"))
        m = {"NRZ": 2, "PAM4": 4, "PAM8": 8}[mod]
        assert len(res.eye_opening["levels"]) == m
        assert res.n_bits > 0
