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
    normalize: bool = True,
) -> go.Figure:
    """Overlaid 2-T eye diagram for the requested quadrature branches.

    The signal is normalised to unit peak envelope so the eye fills the plot
    regardless of the optical power arriving at the receiver (a µW-level ADC
    output would otherwise render as a barely visible flat line).
    """
    period = 2 * sps
    n = len(signal)
    frames = n // period * period
    eye = signal[:frames].reshape(-1, period)
    if normalize and eye.size:
        peak = float(np.percentile(np.abs(signal), 99.0))
        if peak > 0.0:
            eye = eye / peak
    t = np.arange(period) / sps
    fig = go.Figure()
    for branch in branches:
        data = eye.real if branch == "I" else eye.imag
        if eye.shape[0] > max_traces:
            stride = int(np.ceil(eye.shape[0] / max_traces))
            data = data[::stride]
        tt = np.tile(t, data.shape[0])
        color = "#2ca02c" if branch == "I" else "#d62728"
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=data.ravel(),
                mode="lines",
                line={"width": 0.7, "color": color},
                opacity=0.3,
                name=branch,
                showlegend=False,
            )
        )
    # guide lines at the nominal sampling instants (mid-symbol)
    for x_pos in (0.5, 1.5):
        fig.add_vline(x=x_pos, line_dash="dot", line_color=_PLOTLY_GREY, opacity=0.6)
    fig.add_hline(y=0.0, line_dash="dash", line_color=_PLOTLY_GREY, opacity=0.5)
    fig.update_layout(**_base_layout(title, "Time / symbol periods", "Amplitude (normalised)"))
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
    ber_sim_post: Sequence[float] | None = None,
    ber_theory_post: Sequence[float] | None = None,
    post_label: str = "post-FEC",
) -> go.Figure:
    """BER vs OSNR waterfall against the theoretical AWGN limit.

    ``ber_*_post`` add post-FEC series (simulation and/or theory) so the
    coding-gain waterfall drop is visible next to the raw curves.
    """
    fig = go.Figure()
    safe = [max(b, 1e-15) for b in ber_sim]
    fig.add_trace(
        go.Scatter(
            x=list(osnr_db),
            y=[np.log10(b) for b in safe],
            mode="markers+lines",
            name="simulation (pre-FEC)",
            marker={"size": 8, "color": "#1f77b4"},
            line={"color": "#1f77b4", "dash": "solid"},
        )
    )
    if ber_sim_post is not None:
        fig.add_trace(
            go.Scatter(
                x=list(osnr_db),
                y=[np.log10(max(b, 1e-15)) for b in ber_sim_post],
                mode="markers+lines",
                name=f"simulation ({post_label})",
                marker={"size": 8, "color": "#2ca02c"},
                line={"color": "#2ca02c", "dash": "solid"},
            )
        )
    if ber_theory is not None:
        fig.add_trace(
            go.Scatter(
                x=list(osnr_db),
                y=[np.log10(max(b, 1e-15)) for b in ber_theory],
                mode="lines",
                name="AWGN theory (pre-FEC)",
                line={"color": "red", "dash": "dash"},
            )
        )
    if ber_theory_post is not None:
        fig.add_trace(
            go.Scatter(
                x=list(osnr_db),
                y=[np.log10(max(b, 1e-15)) for b in ber_theory_post],
                mode="lines",
                name=f"AWGN theory ({post_label})",
                line={"color": "#9467bd", "dash": "dashdot"},
            )
        )
    fig.update_layout(
        **_base_layout("BER vs OSNR (0.1 nm)", "OSNR (dB)", "log10(BER)"),
        xaxis={"range": [min(osnr_db) - 1.0, max(osnr_db) + 1.0]},
    )
    return fig


def plot_link_profile(
    length_km: float,
    n_spans: int,
    alpha_db_km: float,
    launch_power_dbm: float,
    gamma_per_w_km: float,
    osnr_db: float,
) -> go.Figure:
    """Analytical signal/ASE/nonlinear-phase evolution along the link.

    Shows the familiar EDFA power profile: the signal decays at ``alpha``
    dB/km, an EDFA restores it to the launch power at every span boundary,
    ASE accumulates stage by stage, and the Kerr phase integrates over the
    accumulated effective length. The plot is computed from the link
    parameters (no full SSFM run), so it updates instantly when the sidebar
    sliders move.
    """
    n_spans = max(1, int(n_spans))
    span_km = length_km / n_spans
    alpha_km = alpha_db_km * np.log(10.0) / 10.0  # [1/km]
    p0_w = 10.0 ** (launch_power_dbm / 10.0) / 1000.0  # [W]
    osnr_lin = 10.0 ** (osnr_db / 10.0)
    p_ase_stage_w = p0_w / (n_spans * osnr_lin)  # [W] per amplifier (both pols)

    dz = max(0.05, length_km / 2000.0)
    z = np.arange(0.0, length_km + dz, dz)
    local = z % span_km
    k = np.floor(z / span_km).astype(int)
    p_dbm = launch_power_dbm - alpha_db_km * local
    ase_w = np.maximum(k * p_ase_stage_w, 1e-18)
    ase_dbm = 10.0 * np.log10(ase_w * 1000.0)
    leff_span = (1.0 - np.exp(-alpha_km * span_km)) / alpha_km
    leff_local = (1.0 - np.exp(-alpha_km * local)) / alpha_km
    phi = gamma_per_w_km * p0_w * (k * leff_span + leff_local)

    amp_z = [span_km * i for i in range(1, n_spans)]
    amp_y = [launch_power_dbm] * len(amp_z)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=z,
            y=p_dbm,
            mode="lines",
            name="signal power (dBm)",
            line={"color": "#1f77b4", "width": 2.0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=z,
            y=ase_dbm,
            mode="lines",
            name="accumulated ASE (dBm)",
            line={"color": "#d62728", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=z,
            y=phi,
            mode="lines",
            name="nonlinear phase (rad)",
            yaxis="y2",
            line={"color": "#9467bd", "width": 2.0, "dash": "dash"},
        )
    )
    if amp_z:
        fig.add_trace(
            go.Scatter(
                x=amp_z,
                y=amp_y,
                mode="markers",
                name="EDFA",
                marker={"size": 11, "color": "#2ca02c", "symbol": "triangle-up"},
            )
        )
        for az in amp_z:
            fig.add_vline(x=az, line_dash="dot", line_color="#999999", opacity=0.7)
    fig.add_vline(x=length_km, line_dash="dash", line_color="#333333", opacity=0.8)
    title = "Along the link: power, ASE and nonlinear phase"
    fig.update_layout(
        **_base_layout(title, "Distance (km)", "Power (dBm)"),
        yaxis2={
            "title": "Nonlinear phase (rad)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def plot_penalty_sweep(
    launch_power_dbm: Sequence[float],
    evm_percent: Sequence[float],
    ber: Sequence[float],
    current_power: float,
) -> go.Figure:
    """EVM and BER vs launch power at a fixed OSNR: the nonlinear-penalty curve.

    At low launch power the link is OSNR-limited (flat, AWGN floor); as the
    power rises the Kerr nonlinearity grows and the metrics degrade - the
    classic launch-power penalty.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(launch_power_dbm),
            y=list(evm_percent),
            mode="lines+markers",
            name="EVM (%)",
            line={"color": "#1f77b4", "width": 2.0},
            marker={"size": 7},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(launch_power_dbm),
            y=[np.log10(max(float(b), 1e-12)) for b in ber],
            mode="lines+markers",
            name="log10(BER)",
            yaxis="y2",
            line={"color": "#ff7f0e", "width": 2.0},
            marker={"size": 7},
        )
    )
    fig.add_vline(
        x=current_power,
        line_dash="dot",
        line_color="#333333",
        opacity=0.8,
        annotation_text=f"current: {current_power:.0f} dBm",
        annotation_position="top right",
    )
    title = "Nonlinear penalty vs launch power (same OSNR)"
    fig.update_layout(
        _base_layout(title, "Launch power (dBm)", "EVM (%)"),
        yaxis2={"title": "log10(BER)", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def plot_convergence(err_history: Sequence[float], use_log: bool = True) -> go.Figure:
    """Mean absolute adaptation-error vs symbol index (equalizer).

    ``err_history`` holds one mean-|e| value per adaptation block (a few
    tens of symbols), so the curve shows the CMA acquisition transient
    followed by the MMA/DD steady state.
    """
    fig = go.Figure()
    y = list(err_history)
    if use_log:
        y = [max(float(v), 1e-12) for v in y]
        y = [np.log10(v) for v in y]
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y))),
            y=y,
            mode="lines",
            name="adaptation error",
            line={"color": "#ff7f0e", "width": 1.5},
        )
    )
    ylabel = "log10(mean |e|)" if use_log else "mean |e|"
    fig.update_layout(**_base_layout("Equalizer convergence", "Adaptation block", ylabel))
    return fig
