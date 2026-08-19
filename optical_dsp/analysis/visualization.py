"""Plotly figure builders for the simulation dashboard and notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.signal import welch

_PLOTLY_GREY = "#999999"


def _base_layout(title: str, xlabel: str, ylabel: str) -> dict[str, object]:
    return {
        "title": title,
        "xaxis_title": xlabel,
        "yaxis_title": ylabel,
        "height": 420,
        "template": "plotly_white",
        "font": {"size": 12},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 50},
    }


def plot_constellation(
    samples: NDArray[np.complex128],
    title: str = "Constellation diagram",
    ref_symbols: NDArray[np.complex128] | None = None,
    n_show: int = 8192,
) -> go.Figure:
    """Scatter plot of received (recovered) symbols against the constellation."""
    fig = go.Figure()
    x = samples[:n_show].real
    y = samples[:n_show].imag
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker={"size": 3, "color": "#1f77b4", "opacity": 0.35},
            name="received",
        )
    )
    if ref_symbols is not None:
        fig.add_trace(
            go.Scattergl(
                x=ref_symbols.real,
                y=ref_symbols.imag,
                mode="markers",
                marker={"size": 9, "color": "red", "symbol": "x"},
                name="ideal",
            )
        )
    fig.update_layout(
        **_base_layout(title, "In-phase", "Quadrature"), xaxis={"scaleanchor": "y", "scaleratio": 1}
    )
    return fig


def plot_eye(
    signal: NDArray[np.complex128],
    sps: int,
    title: str = "Eye diagram",
    branches: Sequence[str] = ("I", "Q"),
    max_traces: int = 600,
) -> go.Figure:
    """Overlaid 2-T eye diagram for the requested quadrature branches."""
    period = 2 * sps
    n = len(signal)
    frames = n // period * period
    eye = signal[:frames].reshape(-1, period)
    t = np.arange(period) / sps
    fig = go.Figure()
    for branch in branches:
        data = eye.real if branch == "I" else eye.imag
        if eye.shape[0] > max_traces:
            stride = int(np.ceil(eye.shape[0] / max_traces))
            data = data[::stride]
        tt = np.tile(t, data.shape[0])
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=data.ravel(),
                mode="lines",
                line={"width": 0.4, "color": "#2ca02c" if branch == "I" else "#d62728"},
                opacity=0.25,
                showlegend=False,
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color=_PLOTLY_GREY, opacity=0.5)
    fig.update_layout(**_base_layout(title, "Time / symbol periods", "Amplitude (a.u.)"))
    return fig


def plot_psd(
    signal: NDArray[np.complex128],
    sample_rate: float,
    title: str = "Optical power spectral density",
    axis: str = "GHz",
) -> go.Figure:
    """Welch-estimated PSD of a complex baseband signal."""
    unit = {"GHz": 1e9, "MHz": 1e6, "Hz": 1.0}[axis]
    f, pxx = welch(signal, fs=sample_rate, nperseg=min(len(signal), 4096), return_onesided=False)
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=f / unit,
            y=10.0 * np.log10(np.maximum(pxx, 1e-30)),
            mode="lines",
            line={"color": "#1f77b4", "width": 1.2},
            name="PSD",
        )
    )
    fig.update_layout(**_base_layout(title, f"Frequency ({axis})", "PSD (dB/Hz)"))
    return fig


def plot_waterfall(
    osnr_db: Sequence[float],
    ber_sim: Sequence[float],
    ber_theory: Sequence[float] | None = None,
) -> go.Figure:
    """BER vs OSNR waterfall against the theoretical AWGN limit."""
    fig = go.Figure()
    safe = [max(b, 1e-12) for b in ber_sim]
    fig.add_trace(
        go.Scatter(
            x=list(osnr_db),
            y=[np.log10(b) for b in safe],
            mode="markers+lines",
            name="simulation",
            marker={"size": 8, "color": "#1f77b4"},
            line={"color": "#1f77b4", "dash": "solid"},
        )
    )
    if ber_theory is not None:
        fig.add_trace(
            go.Scatter(
                x=list(osnr_db),
                y=[np.log10(max(b, 1e-12)) for b in ber_theory],
                mode="lines",
                name="AWGN theory",
                line={"color": "red", "dash": "dash"},
            )
        )
    fig.update_layout(
        **_base_layout("BER vs OSNR (0.1 nm)", "OSNR (dB)", "log10(BER)"),
        xaxis={"range": [min(osnr_db) - 1.0, max(osnr_db) + 1.0]},
    )
    return fig


def plot_convergence(err_history: Sequence[float], use_log: bool = True) -> go.Figure:
    """Mean absolute adaptation-error vs symbol index (equalizer)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(err_history))),
            y=list(err_history),
            mode="lines",
            name="adaptation error",
            line={"color": "#ff7f0e", "width": 1.5},
        )
    )
    fig.update_layout(**_base_layout("Equalizer convergence", "Block index", "Mean |e|"))
    return fig
