"""Plotly figure builders for the simulation dashboard and notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.signal import resample_poly, welch

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
    """Scatter plot of received (recovered) symbols against the constellation.

    The received points are power-normalised (unit mean symbol energy) so they
    sit at the same scale as the unit-energy ideal constellation, and the axes
    are locked to a strict 1:1 aspect ratio with fixed limits [-2, 2] so the
    clustering is not squeezed into a strip.
    """
    z = samples[:n_show]
    if len(z):
        z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=z.real,
            y=z.imag,
            mode="markers",
            marker={"size": 4, "color": "#1f77b4", "opacity": 0.4},
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
        **_base_layout(title, "In-phase", "Quadrature"),
        xaxis={"range": [-2, 2], "scaleanchor": "y", "scaleratio": 1},
        yaxis={"range": [-2, 2]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def plot_eye(
    signal: NDArray[np.complex128],
    sps: int,
    title: str = "Eye diagram",
    branches: Sequence[str] = ("I", "Q"),
    max_traces: int = 3000,
    normalize: bool = True,
    target_sps: int = 32,
    opacity: float = 0.08,
) -> go.Figure:
    """Overlaid continuous-trace eye diagram, one subplot per quadrature branch.

    If the signal has fewer than ``target_sps`` samples per symbol it is
    first interpolated with a bandlimited polyphase-sinc resampler (16-32
    samples/symbol), so the transitions are smooth rather than a handful of
    discrete points. The waveform is then sliced into 2-symbol windows centred
    on the sampling instant and each window is drawn as a continuous low-alpha
    line (``opacity`` ~ 0.05-0.1); thousands of overlapping windows accumulate
    into the classic eye shape. Each branch is emitted as a single WebGL trace
    with NaN gaps between windows. The decision instant sits at ``t = 1.0``
    symbol period, the x-axis spans strictly [0, 2], and the amplitude is
    normalised to the decision-instant level so the rails sit at ~±1.
    """
    from plotly.subplots import make_subplots

    if sps < target_sps:
        up = int(np.ceil(target_sps / sps))
        signal = resample_poly(signal, up, 1)
        sps = sps * up
    period = 2 * sps
    n = len(signal) // sps * sps
    if n < 3 * sps:
        pad = np.zeros(sps, dtype=np.complex128)
        signal = np.concatenate([pad, signal, pad])
        n = len(signal) // sps * sps
    grid = signal[:n].reshape(-1, sps)
    off = int(np.argmax(np.var(grid.real, axis=0) + np.var(grid.imag, axis=0)))
    if normalize:
        # scale by the decision-instant amplitude (not the transition
        # overshoot), so the rails land at ~+-1 and the eye always fills
        # the plot regardless of received optical power
        dec = grid[:, off]
        peak = float(np.percentile(np.abs(dec), 99.0))
        if peak > 0.0:
            signal = signal / peak
    centers = np.arange(sps + off, n - sps + 1, sps)
    stride = max(1, int(np.ceil(centers.size / max_traces)))
    centers = centers[::stride]
    t = np.arange(period) / sps

    colors = {"I": "#1f77b4", "Q": "#d62728"}
    fig = make_subplots(
        rows=1,
        cols=len(branches),
        subplot_titles=[f"{b} branch" for b in branches],
        shared_yaxes=True,
    )
    for idx, branch in enumerate(branches, start=1):
        data = signal.real if branch == "I" else signal.imag
        c = colors.get(branch, "#1f77b4")
        x = np.empty(centers.size * (period + 1))
        y = np.empty_like(x)
        for k, center in enumerate(centers):
            base = k * (period + 1)
            x[base : base + period] = t
            x[base + period] = np.nan
            y[base : base + period] = data[center - sps : center + sps]
            y[base + period] = np.nan
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                line={"color": c, "width": 0.9},
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
        fig.add_vline(
            x=1.0,
            line_dash="dot",
            line_color="#111111",
            opacity=0.7,
            row=1,
            col=idx,
        )
        fig.add_vline(x=0.5, line_dash="dot", line_color=_PLOTLY_GREY, opacity=0.45, row=1, col=idx)
        fig.add_vline(x=1.5, line_dash="dot", line_color=_PLOTLY_GREY, opacity=0.45, row=1, col=idx)
    fig.add_hline(y=0.0, line_dash="dash", line_color=_PLOTLY_GREY, opacity=0.5)
    fig.update_layout(
        title=title,
        height=440,
        template="plotly_white",
        font={"size": 12},
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time / symbol periods", range=[0.0, 2.0], row=1, col=1)
    fig.update_xaxes(range=[0.0, 2.0], row=1, col=2)
    fig.update_yaxes(title_text="Amplitude (normalised)", row=1, col=1)
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


def plot_optical_spectrum(
    tx_field: NDArray[np.complex128],
    rx_field: NDArray[np.complex128],
    post_filtered: NDArray[np.complex128],
    sample_rate: float,
    title: str = "Optical / electrical spectrum (PSD)",
) -> go.Figure:
    """Welch PSD in dBm/GHz of TX, RX(+ASE) and post-matched-filter signals.

    ``tx_field`` / ``rx_field`` are the (baseband) optical fields launched and
    received, ``post_filtered`` the electrical signal after the matched
    filter; all must share ``sample_rate``. The received curve sits above the
    TX curve where ASE noise has been added, and the post-filter curve is the
    in-band electrical spectrum the DSP actually processes.
    """
    fig = go.Figure()
    series = (
        (tx_field, "TX (transmitted optical)", "#1f77b4"),
        (rx_field, "RX (received optical + ASE)", "#d62728"),
        (post_filtered, "post-matched filter (electrical)", "#2ca02c"),
    )
    for sig, name, color in series:
        if len(sig) == 0:
            continue
        f, pxx = welch(sig, fs=sample_rate, nperseg=min(len(sig), 8192), return_onesided=False)
        f = np.fft.fftshift(f)
        pxx = np.fft.fftshift(pxx)
        fig.add_trace(
            go.Scatter(
                x=f / 1e9,
                y=10.0 * np.log10(np.maximum(pxx, 1e-30)) + 120.0,
                mode="lines",
                line={"color": color, "width": 1.2},
                name=name,
            )
        )
    fig.update_layout(
        **_base_layout(title, "Frequency (GHz)", "PSD (dBm/GHz)"),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def plot_phase_tracking(
    phase_deg: NDArray[np.float64] | Sequence[float],
    block_centers: NDArray[np.float64] | Sequence[float],
    symbol_rate: float,
    title: str = "Carrier phase recovery tracking (BPS)",
) -> go.Figure:
    """Unwrapped BPS phase estimate vs symbol index.

    The trace shows the laser phase-noise random walk; any linear slope is a
    residual frequency offset, reported in the annotation
    (``slope = 360 deg/symbol`` = one full rotation per symbol = symbol rate).
    """
    x = np.asarray(block_centers, dtype=np.float64)
    y = np.asarray(phase_deg, dtype=np.float64)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": "#ff7f0e", "width": 1.2},
            marker={"size": 3, "color": "#ff7f0e"},
            name="BPS phase estimate",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#999999", opacity=0.5)
    if len(x) >= 2:
        slope = float(np.polyfit(x, y, 1)[0])  # deg/symbol
        freq_mhz = slope / 360.0 * symbol_rate / 1e6
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.05,
            text=f"residual frequency offset ≈ {freq_mhz:+.1f} MHz",
            showarrow=False,
            font={"size": 12},
        )
    fig.update_layout(**_base_layout(title, "Symbol index", "Phase estimate (deg)"))
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

    Shows the realistic EDFA power profile: the signal power decays
    exponentially at ``alpha`` dB/km (a straight ramp on the dBm axis), then a
    discrete gain step at every span boundary restores it to the launch power
    (a crisp vertical jump, drawn as a genuine step). ASE accumulates stage by
    stage, and the Kerr phase integrates over the accumulated effective
    length. The plot is computed from the link parameters (no full SSFM run),
    so it updates instantly when the sidebar sliders move.
    """
    n_spans = max(1, int(n_spans))
    span_km = length_km / n_spans
    alpha_km = alpha_db_km * np.log(10.0) / 10.0  # [1/km]
    p0_w = 10.0 ** (launch_power_dbm / 10.0) / 1000.0  # [W]
    osnr_lin = 10.0 ** (osnr_db / 10.0)
    p_ase_stage_w = p0_w / (n_spans * osnr_lin)  # [W] per amplifier (both pols)

    dz = max(0.05, length_km / 2000.0)

    # Signal power: per-span exponential decay with a vertical EDFA step at
    # every boundary. Duplicate the boundary sample so the jump is a true
    # discrete gain step (bottom of one span -> top of the next) and the
    # sawtooth pattern is unmistakable.
    z_seg, p_seg = [np.array([0.0])], [np.array([launch_power_dbm])]
    for s in range(n_spans):
        a = s * span_km
        b = (s + 1) * span_km
        ramp = np.arange(a + dz, b + dz, dz)
        ramp = np.clip(ramp, None, b)
        z_seg.append(np.concatenate([[a], ramp]))
        decay = launch_power_dbm - alpha_db_km * (ramp - a)
        p_seg.append(np.concatenate([[launch_power_dbm], decay]))
    z = np.concatenate(z_seg)
    p_dbm = np.concatenate(p_seg)

    # Accumulated ASE staircase (constant within a span, steps up at each EDFA).
    k = np.floor(z / span_km).astype(int)
    ase_w = np.maximum(k * p_ase_stage_w, 1e-18)
    ase_dbm = 10.0 * np.log10(ase_w * 1000.0)

    leff_span = (1.0 - np.exp(-alpha_km * span_km)) / alpha_km
    leff_local = (1.0 - np.exp(-alpha_km * (z % span_km))) / alpha_km
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


def plot_convergence(
    err_history: Sequence[float],
    use_log: bool = True,
    switch_at: int | None = None,
) -> go.Figure:
    """Blind equalizer adaptation error vs symbol index.

    Each point is the mean absolute adaptation error over a block of 64
    symbols. The curve falls from the CMA acquisition transient toward a
    steady-state floor set by the residual noise (the same noise that shows
    up as EVM); ``switch_at`` marks the CMA→MMA handover used for
    higher-order QAM.
    """
    fig = go.Figure()
    y = list(err_history)
    if use_log:
        y = [max(float(v), 1e-12) for v in y]
        y = [np.log10(v) for v in y]
    x = list(range(len(y)))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="adaptation error",
            line={"color": "#ff7f0e", "width": 1.5},
            marker={"size": 4, "color": "#ff7f0e"},
        )
    )
    if switch_at is not None and 0 < switch_at < len(y):
        fig.add_vline(
            x=switch_at,
            line_dash="dot",
            line_color="#2ca02c",
            opacity=0.8,
            annotation_text="CMA→MMA",
            annotation_position="top left",
        )
    if y:
        fig.add_hline(
            y=y[-1],
            line_dash="dash",
            line_color="#999999",
            opacity=0.7,
            annotation_text="steady-state floor",
            annotation_position="bottom right",
        )
    ylabel = "log10(mean |e|)" if use_log else "mean |e|"
    title = "Equalizer adaptation error"
    fig.update_layout(**_base_layout(title, "Block index (64 symbols)", ylabel))
    return fig


def plot_imdd_eye(
    signal: NDArray[np.float64],
    sps: int,
    title: str = "IM/DD eye diagram",
    thresholds: Sequence[float] | None = None,
    eye_metrics: dict[str, object] | None = None,
    max_traces: int = 4000,
    target_sps: int = 32,
    opacity: float = 0.1,
) -> go.Figure:
    """Multi-level (intensity) eye diagram with decision-threshold guides.

    The photocurrent is mean-normalised so the PAM rails sit at their average
    optical-power levels; the sampling instant is the column of maximum
    variance. Decision thresholds (midpoints between adjacent measured rails)
    are drawn as dashed horizontal lines, and the EOP / eye-linearity metrics
    are reported in the title when ``eye_metrics`` is supplied.
    """
    if sps < target_sps:
        up = int(np.ceil(target_sps / sps))
        signal = resample_poly(signal, up, 1)
        sps = sps * up
    period = 2 * sps
    n = len(signal) // sps * sps
    grid = signal[:n].reshape(-1, sps)
    off = int(np.argmax(np.var(grid, axis=0)))
    mean_val = float(np.mean(signal)) if len(signal) else 1.0
    norm: NDArray[np.float64] = signal / max(mean_val, 1e-30)

    centers = np.arange(sps + off, n - sps + 1, sps)
    stride = max(1, int(np.ceil(centers.size / max_traces)))
    centers = centers[::stride]
    t = np.arange(period) / sps
    x = np.empty(centers.size * (period + 1))
    y = np.empty_like(x)
    for k, center in enumerate(centers):
        base = k * (period + 1)
        x[base : base + period] = t
        x[base + period] = np.nan
        y[base : base + period] = norm[center - sps : center + sps]
        y[base + period] = np.nan

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            line={"color": "#1f77b4", "width": 0.9},
            opacity=opacity,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_vline(x=1.0, line_dash="dot", line_color="#111111", opacity=0.7)
    fig.add_vline(x=0.5, line_dash="dot", line_color=_PLOTLY_GREY, opacity=0.45)
    fig.add_vline(x=1.5, line_dash="dot", line_color=_PLOTLY_GREY, opacity=0.45)
    th_list = list(thresholds) if thresholds else []
    ymax = 0.0
    for th in th_list:
        fig.add_hline(y=th, line_dash="dash", line_color="#d62728", opacity=0.8)
        ymax = max(ymax, float(th))
    yrange = [0.0, max(1.2 * ymax, 1.1)] if th_list else [0.0, 1.6]

    if eye_metrics:
        eop = eye_metrics.get("eop_db")
        lin = eye_metrics.get("eye_linearity")
        if eop is not None and lin is not None:
            title = (
                f"{title}<br><sup>EOP {eop:.1f} dB · eye linearity {lin:.2f} · M-level eye</sup>"
            )
    fig.update_layout(
        title=title,
        height=440,
        template="plotly_white",
        font={"size": 12},
        margin={"l": 60, "r": 20, "t": 80, "b": 50},
        showlegend=False,
        xaxis={"title": "Time / symbol periods", "range": [0.0, 2.0]},
        yaxis={"title": "Normalised photocurrent", "range": yrange},
    )
    return fig


def plot_sensitivity_waterfall(
    rop_dbm: Sequence[float],
    curves: dict[str, Sequence[float]],
    crossings: dict[str, float] | None = None,
    target_ber: float = 1e-3,
    title: str = "Receiver sensitivity: BER vs optical power",
) -> go.Figure:
    """BER vs received optical power (ROP) for each receiver type.

    Semi-log plot of the simulated sensitivity waterfall with a dashed target
    line and vertical markers at the power where each receiver first crosses
    ``target_ber``. PIN and APD curves are coloured blue and red.
    """
    colors = {"PIN": "#1f77b4", "APD": "#d62728"}
    fig = go.Figure()
    for rx, ber_curve in curves.items():
        safe = [max(float(b), 1e-15) for b in ber_curve]
        fig.add_trace(
            go.Scatter(
                x=list(rop_dbm),
                y=safe,
                mode="markers+lines",
                name=f"{rx} (simulated)",
                marker={"size": 7, "color": colors.get(rx, "#1f77b4")},
                line={"color": colors.get(rx, "#1f77b4"), "width": 1.8},
            )
        )
        cross = crossings.get(rx) if crossings else None
        if cross is not None and np.isfinite(cross):
            fig.add_vline(
                x=float(cross),
                line_dash="dot",
                line_color=colors.get(rx, "#1f77b4"),
                opacity=0.7,
                annotation_text=f"{rx} {cross:.1f} dBm",
                annotation_position="top left",
            )
    fig.add_hline(
        y=target_ber,
        line_dash="dash",
        line_color="#333333",
        opacity=0.8,
        annotation_text=f"target {target_ber:.0e}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        **_base_layout(title, "Received optical power (dBm)", "BER"),
        yaxis={"type": "log", "range": [-15.0, 0.0]},
        xaxis={"range": [min(rop_dbm) - 1.0, max(rop_dbm) + 1.0]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def plot_link_budget_bar(
    budget: Sequence[tuple[str, float | None, float]],
    title: str = "IM/DD link budget",
) -> go.Figure:
    """Waterfall bar chart of the link budget stages.

    ``budget`` items are ``(label, increment_db, cumulative_db)`` where
    ``increment_db is None`` marks an absolute ("total") bar: transmitter
    power, received power and receiver sensitivity. Relative bars (connector,
    fibre, splitter losses and system margin) chain onto the running total so
    the waterfall lands on the sensitivity target.
    """
    labels = [item[0] for item in budget]
    measures = [
        "total" if i == 0 or item[1] is None else "relative" for i, item in enumerate(budget)
    ]
    values = [0.0 if item[1] is None else float(item[1]) for item in budget]
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}},
            connector={"line": {"color": "#999999", "dash": "dot", "width": 1}},
            text=[f"{v:.1f} dB" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout(title, "Link budget stage", "Power / loss (dB)"),
        showlegend=False,
    )
    return fig
