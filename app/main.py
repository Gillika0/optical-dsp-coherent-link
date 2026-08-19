"""Streamlit dashboard for the coherent optical link & DSP simulator.

Run with::

    streamlit run app/main.py

The sidebar configures the link; the main pane shows metrics and the
recovered-signal visualizations produced by ``run_link``.
"""

from __future__ import annotations

import warnings

import numpy as np
import streamlit as st
from optical_dsp.analysis.metrics import (
    HD_FEC_RS255_239,
    STRONG_FEC_RS255_213,
    apply_fec,
    evm_to_snr_db,
    q_factor_from_ber,
    theoretical_ber_qam,
)
from optical_dsp.analysis.visualization import (
    plot_constellation,
    plot_convergence,
    plot_eye,
    plot_link_profile,
    plot_penalty_sweep,
    plot_psd,
    plot_waterfall,
)
from optical_dsp.pipeline import LinkConfig, LinkResult, run_link
from optical_dsp.utils import get_constellation, ref_bandwidth_hz

warnings.filterwarnings("ignore")


def _build_config(
    modulation: str,
    symbol_rate_gbd: float,
    length_km: float,
    launch_power_dbm: float,
    osnr_db: float,
    tx_linewidth_khz: float,
    lo_linewidth_khz: float,
    lo_offset_ghz: float,
    run_cdc: bool,
    equalizer_taps: int,
    mu_mma: float,
    bps_phases: int,
    n_symbols: int,
    seed: int,
    n_spans: int,
    fec: str,
) -> LinkConfig:
    return LinkConfig(
        modulation=modulation,
        symbol_rate=symbol_rate_gbd * 1e9,
        length_km=length_km,
        launch_power_dbm=launch_power_dbm,
        osnr_db=osnr_db,
        tx_linewidth_khz=tx_linewidth_khz,
        lo_linewidth_khz=lo_linewidth_khz,
        lo_freq_offset_ghz=lo_offset_ghz,
        run_cdc=run_cdc,
        equalizer_taps=equalizer_taps,
        mu_mma=mu_mma,
        bps_phases=bps_phases,
        n_symbols=n_symbols,
        seed=seed,
        n_spans=n_spans,
        fec=fec,
    )


@st.cache_data(show_spinner=False)
def _run_cached(cfg: LinkConfig) -> LinkResult:
    return run_link(cfg, quiet=True)


@st.cache_data(show_spinner=False)
def _waterfall_cached(
    modulation: str,
    symbol_rate_gbd: float,
    length_km: float,
    seed: int,
    n_spans: int,
    fec: str,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    grids = {
        "QPSK": np.linspace(10.0, 18.0, 6),
        "16-QAM": np.linspace(16.0, 26.0, 6),
        "64-QAM": np.linspace(20.0, 30.0, 6),
    }
    osnr_grid = grids[modulation]
    const = get_constellation(modulation)
    ber_sim: list[float] = []
    ber_theory: list[float] = []
    ber_sim_post: list[float] = []
    ber_theory_post: list[float] = []
    fec_code = None
    if fec == "hd":
        fec_code = HD_FEC_RS255_239
    elif fec == "strong":
        fec_code = STRONG_FEC_RS255_213
    for osnr_db in osnr_grid:
        cfg = LinkConfig(
            modulation=modulation,
            n_symbols=2**12,
            length_km=length_km,
            osnr_db=float(osnr_db),
            seed=seed,
            n_spans=n_spans,
            fec=fec,
        )
        r = run_link(cfg, quiet=True)
        ber_sim.append(r.ber.ber if r.ber is not None else 1.0)
        # Per-polarisation SNR from OSNR: the amplifier scales the per-pol ASE
        # so that n0 = Ps / (OSNR * B_ref); after the matched filter the per-pol
        # symbol SNR is Es/N0 = OSNR * B_ref / Rs (no extra 3 dB).
        ref_bw = ref_bandwidth_hz(1550.0e-9, 0.1)
        snr_db = float(osnr_db + 10.0 * np.log10(ref_bw / (symbol_rate_gbd * 1e9)))
        theory = theoretical_ber_qam(snr_db, const.order)
        ber_theory.append(theory)
        if fec_code is not None:
            ber_sim_post.append(apply_fec(ber_sim[-1], fec_code))
            ber_theory_post.append(apply_fec(theory, fec_code))
        else:
            ber_sim_post.append(ber_sim[-1])
            ber_theory_post.append(theory)
    return (
        [float(osnr_db) for osnr_db in osnr_grid],
        ber_sim,
        ber_theory,
        ber_sim_post,
        ber_theory_post,
    )


@st.cache_data(show_spinner=False)
def _launch_sweep_cached(
    modulation: str,
    symbol_rate_gbd: float,
    length_km: float,
    osnr_db: float,
    seed: int,
    n_spans: int,
) -> tuple[list[float], list[float], list[float]]:
    """Sweep the launch power at a fixed OSNR to expose the nonlinear penalty.

    A lighter sweep (fewer symbols) since it only needs the trend, not a
    precise BER.
    """
    powers = list(range(-8, 13, 4))  # -8, -4, 0, 4, 8, 12 dBm
    evm: list[float] = []
    ber: list[float] = []
    for p_dbm in powers:
        cfg = LinkConfig(
            modulation=modulation,
            symbol_rate=symbol_rate_gbd * 1e9,
            length_km=length_km,
            osnr_db=osnr_db,
            launch_power_dbm=float(p_dbm),
            seed=seed,
            n_spans=n_spans,
            n_symbols=2**11,
            fec="none",
        )
        r = run_link(cfg, quiet=True)
        evm.append(float(max(r.evm_percent)))
        ber.append(r.ber.ber if r.ber is not None else 1.0)
    return [float(p) for p in powers], evm, ber


def _sidebar() -> LinkConfig:
    st.sidebar.title("Link configuration")
    modulation = st.sidebar.selectbox("Modulation", ["QPSK", "16-QAM", "64-QAM"], index=0)
    symbol_rate = st.sidebar.number_input(
        "Symbol rate (GBd)", min_value=4.0, max_value=128.0, value=32.0, step=4.0
    )
    length_km = st.sidebar.slider("Link length (km)", 5.0, 200.0, 80.0, step=5.0)
    n_spans = st.sidebar.slider("Fibre spans / amplifiers", 1, 16, 1, step=1)
    st.sidebar.caption(
        "Each span is followed by an EDFA restoring the launch power; more "
        "spans accumulate more nonlinear phase for the same end OSNR."
    )
    launch_power_dbm = st.sidebar.slider("Launch power (dBm)", -10.0, 12.0, 0.0, step=1.0)
    osnr_db = st.sidebar.slider("Target OSNR (dB, 0.1 nm)", 8.0, 40.0, 22.0, step=1.0)
    tx_linewidth = st.sidebar.slider("TX linewidth (kHz)", 0.0, 2000.0, 100.0, step=10.0)
    lo_linewidth = st.sidebar.slider("LO linewidth (kHz)", 0.0, 2000.0, 100.0, step=10.0)
    lo_offset = st.sidebar.slider("LO frequency offset (GHz)", -1.0, 1.0, 0.0, step=0.05)
    st.sidebar.markdown("---")
    st.sidebar.subheader("DSP")
    enable_cdc = st.sidebar.checkbox("Chromatic-dispersion compensation", value=True)
    eq_taps = st.sidebar.slider("Equalizer taps", 5, 31, 15, step=2)
    mu_mma = st.sidebar.select_slider(
        "DD/MMA step size (mu)", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3
    )
    bps_phases = st.sidebar.slider("BPS phases / quadrant", 8, 128, 32, step=8)
    n_log2 = st.sidebar.slider("Symbols (2^n)", 10, 15, 12, step=1)
    seed = st.sidebar.number_input("Seed", value=1234, step=1)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Forward error correction")
    fec = st.sidebar.selectbox(
        "FEC",
        ["none", "hd", "strong"],
        format_func=lambda v: {
            "none": "Off",
            "hd": "HD-FEC (7%, RS(255,239))",
            "strong": "Strong FEC (20%, RS(255,213))",
        }[v],
        index=0,
    )
    st.sidebar.caption(
        "Post-FEC BER is modelled with bounded-distance hard-decision decoding "
        "of the RS code from the simulated (pre-FEC) BER."
    )

    return _build_config(
        modulation=modulation,
        symbol_rate_gbd=float(symbol_rate),
        length_km=float(length_km),
        launch_power_dbm=float(launch_power_dbm),
        osnr_db=float(osnr_db),
        tx_linewidth_khz=float(tx_linewidth),
        lo_linewidth_khz=float(lo_linewidth),
        lo_offset_ghz=float(lo_offset),
        run_cdc=enable_cdc,
        equalizer_taps=eq_taps,
        mu_mma=float(mu_mma),
        bps_phases=bps_phases,
        n_symbols=2**n_log2,
        seed=int(seed),
        n_spans=int(n_spans),
        fec=fec,
    )


def _show_metrics(res: LinkResult) -> None:
    c = res.config
    if res.ber is None:
        return
    evm0, evm1 = res.evm_percent
    ber = res.ber.ber
    post = res.post_fec_ber
    n_bits = max(res.ber.n_bits, 1)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("EVM X (%)", f"{evm0:.1f}")
    col2.metric("EVM Y (%)", f"{evm1:.1f}")
    if ber > 0.0:
        col3.metric("BER (pre-FEC)", f"{ber:.2e}")
        q_pre = q_factor_from_ber(ber, cap_db=30.0)
    else:
        col3.metric("BER (pre-FEC)", f"< {1 / n_bits:.1e}")
        q_pre = q_factor_from_ber(1 / n_bits, cap_db=30.0)
    if post is not None:
        if post <= 0.0:
            col4.metric("BER (post-FEC)", "< 1e-15")
            q_post = q_factor_from_ber(1e-15, cap_db=30.0)
        else:
            col4.metric("BER (post-FEC)", f"{post:.1e}")
            q_post = q_factor_from_ber(post, cap_db=30.0)
        col5.metric("Q-factor (dB)", f">{q_post:.0f} dB")
        st.caption(
            f"{res.extra.get('fec', 'FEC')} | "
            f"FOE: {res.extra['freq_offset_est_hz'] / 1e9:.3f} GHz | "
            f"SNR: {evm_to_snr_db((evm0 + evm1) / 2):.1f} dB"
        )
    else:
        col4.metric("FEC", "off")
        col5.metric("Q-factor (dB)", f">{q_pre:.0f} dB")
        st.caption(
            f"FOE: {res.extra['freq_offset_est_hz'] / 1e9:.3f} GHz | "
            f"SNR: {evm_to_snr_db((evm0 + evm1) / 2):.1f} dB"
        )
    st.caption(
        f"{c.modulation} @ {c.symbol_rate / 1e9:.0f} GBd, {c.length_km:.0f} km, "
        f"{c.n_spans} span(s), launch {c.launch_power_dbm:.0f} dBm"
    )


def _constellation_tabs(res: LinkResult) -> None:
    st.header("Recovered constellations")
    tab_x, tab_y = st.tabs(["Polarisation X", "Polarisation Y"])
    z = res.cr_out
    const = get_constellation(res.config.modulation)
    ref = np.tile(const.symbols, len(z) // const.order)
    with tab_x:
        st.plotly_chart(
            plot_constellation(z[:, 0], title="X-pol after BPS", ref_symbols=ref),
            use_container_width=True,
        )
    with tab_y:
        st.plotly_chart(
            plot_constellation(z[:, 1], title="Y-pol after BPS", ref_symbols=ref),
            use_container_width=True,
        )


def _eye_psd_tabs(res: LinkResult) -> None:
    st.header("Eye diagrams (pre vs post DSP)")
    tab_x, tab_y, tab_psd = st.tabs(["Polarisation X", "Polarisation Y", "Power spectrum"])
    sps = res.config.sps
    with tab_x:
        c_pre, c_post = st.columns(2)
        c_pre.plotly_chart(
            plot_eye(res.eye_pre[:, 0], sps, title="Pre-DSP eye (ADC output)"),
            use_container_width=True,
        )
        c_post.plotly_chart(
            plot_eye(res.eye_post[:, 0], sps, title="Post-DSP eye (CDC + matched filter)"),
            use_container_width=True,
        )
    with tab_y:
        c_pre, c_post = st.columns(2)
        c_pre.plotly_chart(
            plot_eye(res.eye_pre[:, 1], sps, title="Pre-DSP eye (ADC output)"),
            use_container_width=True,
        )
        c_post.plotly_chart(
            plot_eye(res.eye_post[:, 1], sps, title="Post-DSP eye (CDC + matched filter)"),
            use_container_width=True,
        )
    with tab_psd:
        sample_rate = res.config.symbol_rate * sps
        st.plotly_chart(
            plot_psd(res.rx_wide, sample_rate, title="Received PSD (post-CDC)"),
            use_container_width=True,
        )


def main() -> None:
    """Streamlit entry point (also the ``optical-dsp-demo`` console script)."""
    st.set_page_config(page_title="Coherent Optical Link & DSP Simulator", layout="wide")
    st.title("Coherent Optical Link & DSP Simulator")
    st.caption(
        "PDM QPSK/QAM over a nonlinear SSFM fibre, with a full coherent "
        "receiver DSP chain (CDC, matched filter, MIMO equalizer, FOE, BPS)."
    )

    cfg = _sidebar()
    osnr_grid, ber_sweep, ber_theory, ber_sweep_post, ber_theory_post = _waterfall_cached(
        cfg.modulation,
        cfg.symbol_rate / 1e9,
        cfg.length_km,
        cfg.seed if cfg.seed is not None else 1234,
        cfg.n_spans,
        cfg.fec,
    )

    if st.sidebar.button("Run simulation", type="primary"):
        with st.spinner("Simulating link..."):
            result = _run_cached(cfg)
        st.session_state["result"] = result

    if "result" not in st.session_state:
        st.info("Configure the link on the sidebar and press **Run simulation**.")
        return

    res = st.session_state["result"]
    if res.config != cfg:
        st.warning("Sidebar settings changed since the last run - press Run again.")
    _show_metrics(res)
    _constellation_tabs(res)
    _eye_psd_tabs(res)

    st.header("Equalizer convergence")
    st.plotly_chart(plot_convergence(res.equalizer_errors), use_container_width=True)

    st.header("Waterfall (BER vs OSNR, theory vs simulation)")
    st.plotly_chart(
        plot_waterfall(
            osnr_grid,
            ber_sweep,
            ber_theory,
            ber_sim_post=ber_sweep_post if cfg.fec != "none" else None,
            ber_theory_post=ber_theory_post if cfg.fec != "none" else None,
        ),
        use_container_width=True,
    )
    st.caption(
        "The AWGN curve is the ideal limit for the per-polarisation SNR "
        "Es/N0 = OSNR · B_ref/Rs. The simulation always lands above it: laser "
        "phase noise, ADC quantization, the blind equalizer's steady-state "
        "residual and residual crosstalk cost a few dB, and at very low OSNR "
        "the carrier-recovery stage itself fails, flattening the curve. "
        "When FEC is enabled the green (post-FEC) curves show the coding-gain "
        "waterfall drop: once the pre-FEC BER is inside the code's correction "
        "capability the post-FEC BER collapses toward 1e-15."
    )

    st.header("Along the link (analytical)")
    st.plotly_chart(
        plot_link_profile(
            length_km=cfg.length_km,
            n_spans=cfg.n_spans,
            alpha_db_km=cfg.alpha_db_km,
            launch_power_dbm=cfg.launch_power_dbm,
            gamma_per_w_km=1317.5,  # n2=2.6e-20, Aeff=80 um^2 @ 1550 nm
            osnr_db=cfg.osnr_db,
        ),
        use_container_width=True,
    )
    st.caption(
        "Analytical profile from the sidebar parameters: the signal decays at "
        "0.2 dB/km and is restored to the launch power at every EDFA, the ASE "
        "accumulates stage by stage (the end-of-link OSNR stays fixed because "
        "the budget is split evenly), and the Kerr phase integrates over the "
        "effective length - it grows with the number of spans."
    )

    st.header("Nonlinear penalty vs launch power")
    sweep_powers, sweep_evm, sweep_ber = _launch_sweep_cached(
        cfg.modulation,
        cfg.symbol_rate / 1e9,
        cfg.length_km,
        cfg.osnr_db,
        cfg.seed if cfg.seed is not None else 1234,
        cfg.n_spans,
    )
    st.plotly_chart(
        plot_penalty_sweep(sweep_powers, sweep_evm, sweep_ber, cfg.launch_power_dbm),
        use_container_width=True,
    )
    st.caption(
        "Same OSNR, seed and number of spans: at low launch power the link is "
        "AWGN-limited (flat floor); as the power rises, self-phase and cross-"
        "phase modulation distort the constellation and the metrics degrade. "
        "The dashed line marks the launch power selected on the sidebar."
    )


if __name__ == "__main__":
    main()
