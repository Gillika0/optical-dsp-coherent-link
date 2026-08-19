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
from optical_dsp.analysis.metrics import evm_to_snr_db, q_factor_from_ber, theoretical_ber_qam
from optical_dsp.analysis.visualization import (
    plot_constellation,
    plot_convergence,
    plot_eye,
    plot_psd,
    plot_waterfall,
)
from optical_dsp.pipeline import LinkConfig, LinkResult, run_link
from optical_dsp.utils import get_constellation

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
    )


@st.cache_data(show_spinner=False)
def _run_cached(cfg: LinkConfig) -> LinkResult:
    return run_link(cfg, quiet=True)


@st.cache_data(show_spinner=False)
def _waterfall_cached(
    modulation: str, symbol_rate_gbd: float, length_km: float, seed: int
) -> tuple[list[float], list[float], list[float]]:
    osnr_grid = np.linspace(10.0, 24.0, 8)
    const = get_constellation(modulation)
    ber_sim: list[float] = []
    ber_theory: list[float] = []
    for osnr_db in osnr_grid:
        cfg = LinkConfig(
            modulation=modulation,
            n_symbols=2**11,
            length_km=length_km,
            osnr_db=float(osnr_db),
            seed=seed,
            mu_mma=1e-3,
        )
        r = run_link(cfg, quiet=True)
        ber_sim.append(r.ber.ber if r.ber is not None else 1.0)
        snr_db = osnr_db + 10.0 * np.log10(12.5e9 / (symbol_rate_gbd * 1e9) / 2.0)
        ber_theory.append(theoretical_ber_qam(float(snr_db), const.order))
    return [float(osnr_db) for osnr_db in osnr_grid], ber_sim, ber_theory


def _sidebar() -> LinkConfig:
    st.sidebar.title("Link configuration")
    modulation = st.sidebar.selectbox("Modulation", ["QPSK", "16-QAM", "64-QAM"], index=0)
    symbol_rate = st.sidebar.number_input(
        "Symbol rate (GBd)", min_value=4.0, max_value=128.0, value=32.0, step=4.0
    )
    length_km = st.sidebar.slider("Link length (km)", 5.0, 200.0, 80.0, step=5.0)
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
    )


def _show_metrics(res: LinkResult) -> None:
    c = res.config
    if res.ber is None:
        return
    evm0, evm1 = res.evm_percent
    ber = res.ber.ber
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("EVM X (%)", f"{evm0:.1f}")
    col2.metric("EVM Y (%)", f"{evm1:.1f}")
    col3.metric("BER", f"{ber:.2e}")
    col4.metric("Q-factor (dB)", f"{q_factor_from_ber(ber):.1f}")
    st.caption(
        f"Estimated FOE: {res.extra['freq_offset_est_hz'] / 1e9:.3f} GHz | "
        f"SNR: {evm_to_snr_db((evm0 + evm1) / 2):.1f} dB | "
        f"{c.modulation} @ {c.symbol_rate / 1e9:.0f} GBd, {c.length_km:.0f} km"
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
    st.header("Received signal")
    tab_eye, tab_psd = st.tabs(["Eye diagram", "Power spectrum"])
    sample_rate = res.config.symbol_rate * res.config.sps
    with tab_eye:
        st.plotly_chart(
            plot_eye(res.rx_wide, res.config.sps, title="X-pol eye (post-CDC)"),
            use_container_width=True,
        )
    with tab_psd:
        st.plotly_chart(
            plot_psd(res.rx_wide, sample_rate, title="Received PSD"),
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
    osnr_grid, ber_sweep, ber_theory = _waterfall_cached(
        cfg.modulation,
        cfg.symbol_rate / 1e9,
        cfg.length_km,
        cfg.seed if cfg.seed is not None else 1234,
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
        plot_waterfall(osnr_grid, ber_sweep, ber_theory),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
