"""Eye diagram tests: continuous trace overlay, interpolation, windowing."""

from __future__ import annotations

import numpy as np
from optical_dsp.analysis.visualization import (
    plot_eye,
    plot_imdd_eye,
    plot_link_budget_bar,
    plot_sensitivity_waterfall,
)
from optical_dsp.utils import QPSK


def _qpsk_signal(n_symbols: int, sps: int, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    const = QPSK()
    sym = const.symbols[rng.integers(0, 4, n_symbols)]
    sig = np.repeat(sym, sps).astype(np.complex128)
    if noise:
        sig = sig + noise * (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size))
    return sig


def test_eye_continuous_trace_overlay() -> None:
    sig = _qpsk_signal(1024, 4)
    fig = plot_eye(sig, 4)
    assert len(fig.data) == 2  # I and Q, one WebGL trace each
    for tr in fig.data:
        x = np.asarray(tr.x)
        n_nan = int(np.isnan(x).sum())
        assert n_nan > 100  # many 2-symbol windows as NaN-separated segments
        assert tr.type == "scattergl"
        assert tr.mode == "lines"
        assert tr.opacity < 0.15


def test_eye_decision_instant_centred() -> None:
    sig = _qpsk_signal(1024, 4)
    fig = plot_eye(sig, 4)
    assert list(fig.layout.xaxis.range) == [0.0, 2.0]  # type: ignore[union-attr]
    vlines = {round(float(s.x0), 2) for s in fig.layout.shapes if s.x0 is not None}
    assert 1.0 in vlines  # decision instant


def test_eye_interpolates_low_sps_input() -> None:
    sig = _qpsk_signal(512, 1)  # 1 sample/symbol -> interpolated to 32
    fig = plot_eye(sig, 1)
    x = np.asarray(fig.data[0].x)
    finite = x[~np.isnan(x)]
    assert finite.max() <= 2.0
    # window count: (512 symbols - 2) windows, sampled to max_traces
    n_nan = int(np.isnan(x).sum())
    assert n_nan == len(fig.data[0].x) // 65  # period+1 = 65 samples per window
    assert 0 < n_nan <= 3000


def test_eye_rails_at_decision_instant() -> None:
    # noiseless QPSK: the decision instant must sit on the constellation rails
    sig = _qpsk_signal(2048, 8)
    fig = plot_eye(sig, 8)
    tr = fig.data[0]
    x = np.asarray(tr.x)
    y = np.asarray(tr.y)
    period = 64  # 2 * 32 sps after interpolation
    windows = []
    idx = 0
    while idx < len(x):
        if np.isnan(x[idx]):
            idx += 1
            continue
        windows.append(y[idx : idx + period])
        idx += period + 1
    w = np.stack(windows)
    dec = w[:, period // 2]  # t = 1.0
    pos, neg = dec[dec > 0].mean(), dec[dec < 0].mean()
    assert pos > 0.5 and neg < -0.5  # rails sit near +-1 (normalised)


def _pam_signal(n_symbols: int, sps: int, m: int, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    levels = (np.arange(m) / (m - 1.0)).astype(np.float64)
    sym = levels[rng.integers(0, m, n_symbols)]
    sig = np.repeat(sym, sps).astype(np.float64)
    if noise:
        sig = sig + noise * rng.standard_normal(sig.size)
    return sig


def test_imdd_eye_multi_level_and_thresholds() -> None:
    sig = _pam_signal(1024, 8, 4, noise=0.01)
    fig = plot_imdd_eye(sig, 8, thresholds=[0.333, 0.667], eye_metrics=None)
    assert len(fig.data) == 1  # single intensity trace
    assert fig.data[0].type == "scattergl"
    # one dashed threshold hline per decision line
    hlines = [
        s for s in fig.layout.shapes if s.type == "line" and s.y0 is not None and s.y0 == s.y1
    ]
    assert len(hlines) == 2


def test_imdd_eye_title_annotations() -> None:
    sig = _pam_signal(512, 8, 4, noise=0.02)
    metrics = {"eop_db": 2.5, "eye_linearity": 0.9}
    fig = plot_imdd_eye(sig, 8, thresholds=[0.333, 0.667], eye_metrics=metrics)
    assert "EOP 2.5 dB" in fig.layout.title.text
    assert "0.90" in fig.layout.title.text


def test_sensitivity_waterfall_curves() -> None:
    rop = [-30.0, -25.0, -20.0, -15.0, -10.0]
    curves = {"PIN": [0.4, 0.2, 1e-3, 1e-5, 1e-8], "APD": [0.2, 1e-3, 1e-5, 1e-8, 1e-9]}
    fig = plot_sensitivity_waterfall(rop, curves, crossings={"PIN": -20.0, "APD": -25.0})
    assert len(fig.data) == 2
    assert fig.layout.yaxis.type == "log"  # type: ignore[union-attr]
    # both crossings drawn as vertical shapes
    vlines = [
        s for s in fig.layout.shapes if s.type == "line" and s.x0 is not None and s.x0 == s.x1
    ]
    assert len(vlines) == 2


def test_link_budget_waterfall_bars() -> None:
    budget = [
        ("Transmitter power", 3.0, 3.0),
        ("Connector losses", -1.0, 2.0),
        ("Received power (ROP)", None, -2.0),
        ("Receiver sensitivity", None, -15.0),
    ]
    fig = plot_link_budget_bar(budget)
    tr = fig.data[0]
    assert tr.type == "waterfall"
    assert list(tr.x) == [
        "Transmitter power",
        "Connector losses",
        "Received power (ROP)",
        "Receiver sensitivity",
    ]
    assert list(tr.measure) == ["total", "relative", "total", "total"]
