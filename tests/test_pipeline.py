"""End-to-end pipeline smoke tests."""

from __future__ import annotations

import numpy as np
from optical_dsp.pipeline import LinkConfig, run_link


def test_pipeline_qpsk_healthy() -> None:
    cfg = LinkConfig(
        modulation="QPSK",
        n_symbols=2**12,
        length_km=80.0,
        osnr_db=20.0,
        seed=1234,
        lo_freq_offset_ghz=0.1,
        n_cma=2000,
        bps_phases=32,
        bps_block=16,
    )
    r = run_link(cfg)
    assert np.all(r.evm_percent[0] < 20.0)
    assert r.ber is not None and r.ber.ber < 1e-2
    # the FOE should have found the injected 0.1 GHz detuning
    assert np.isclose(r.extra["freq_offset_est_hz"] / 1e9, 0.1, atol=0.02)


def test_pipeline_16qam_healthy() -> None:
    cfg = LinkConfig(
        modulation="16-QAM",
        n_symbols=2**13,
        length_km=40.0,
        osnr_db=28.0,
        seed=7,
        n_cma=4000,
    )
    r = run_link(cfg)
    assert np.all(r.evm_percent[0] < 15.0)
    assert r.ber is not None and r.ber.ber < 1e-2


def test_pipeline_no_cdc_still_runs() -> None:
    cfg = LinkConfig(
        modulation="QPSK",
        n_symbols=2**12,
        length_km=40.0,
        run_cdc=False,
        seed=3,
        n_cma=2000,
    )
    r = run_link(cfg)
    assert r.extra["n_samples_eq"] > 0


def test_pipeline_multispan_grows_nonlinear_penalty() -> None:
    # same length, OSNR and seed: more spans restore the launch power more
    # often, so the accumulated nonlinear phase (and the EVM) must increase
    evm = []
    for spans in (1, 8):
        cfg = LinkConfig(
            modulation="16-QAM",
            n_symbols=2**12,
            length_km=40.0,
            osnr_db=26.0,
            launch_power_dbm=8.0,
            seed=7,
            n_spans=spans,
        )
        r = run_link(cfg)
        evm.append(float(max(r.evm_percent)))
    assert evm[1] > evm[0]
    assert r.extra["n_spans"] == 8


def test_pipeline_fec_reports_post_ber() -> None:
    cfg = LinkConfig(
        modulation="16-QAM",
        n_symbols=2**12,
        length_km=40.0,
        osnr_db=26.0,
        seed=7,
        fec="hd",
    )
    r = run_link(cfg)
    assert r.ber is not None
    assert r.post_fec_ber is not None
    # inside the HD-FEC capability the post-FEC BER collapses well below raw
    assert r.post_fec_ber < r.ber.ber
    assert r.post_fec_ber < 1e-4
    assert r.extra["fec"] == "HD-FEC RS(255,239) 7%"


def test_pipeline_rejects_unknown_fec() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_link(LinkConfig(n_symbols=2**10, fec="turbo"))
