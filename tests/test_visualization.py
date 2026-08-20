"""Eye diagram tests: continuous trace overlay, interpolation, windowing."""

from __future__ import annotations

import numpy as np
from optical_dsp.analysis.visualization import plot_eye
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
