"""Streamlit page: direct-detection / short-reach (PON) transmission.

Intensity-modulation links (NRZ / PAM4 / PAM8) with DML/EML laser chirp and
extinction ratio, fibre + splitter losses, PIN/APD receivers and baud-spaced
FFE/DFE equalization. Shows pre- and post-equalizer multi-level eyes with
clamped EOP/TDECQ metrics, a BER-vs-ROP receiver-sensitivity waterfall and a
full link-budget chart.
"""

from __future__ import annotations

import warnings

import streamlit as st
from optical_dsp.analysis.visualization import (
    plot_imdd_eye,
    plot_link_budget_bar,
    plot_sensitivity_waterfall,
)
from optical_dsp.imdd import ImddConfig, ImddResult, imdd_sensitivity, run_imdd

warnings.filterwarnings("ignore")

_RECEIVERS = ["PIN", "APD"]
_PRESETS = {
    "O-band PAM4 100G (26.56 GBd, 1310 nm)": {
        "modulation": "PAM4",
        "symbol_rate": 26.56,
        "band": "O-band (1310 nm)",
        "length": 20.0,
        "laser": "EML",
    },
    "C-band 10G NRZ (10 GBd, 1550 nm, 20 km)": {
        "modulation": "NRZ",
        "symbol_rate": 10.0,
        "band": "C-band (1550 nm)",
        "length": 20.0,
        "laser": "EML",
    },
}
_BANDS = ["O-band (1310 nm)", "C-band (1550 nm)"]


def _build_config(
    modulation: str,
    symbol_rate_gbd: float,
    sps: int,
    laser_type: str,
    extinction_ratio_db: float,
    chirp_alpha: float,
    length_km: float,
    alpha_db_km: float,
    wavelength_nm: float,
    dispersion_ps: float,
    connector_loss_db: float,
    splitter_ratio: int,
    tx_power_dbm: float,
    receiver_type: str,
    apd_gain: float,
    rx_bw_ghz: float,
    equalizer_type: str,
    equalizer_taps: int,
    n_symbols: int,
    seed: int,
) -> ImddConfig:
    return ImddConfig(
        modulation=modulation,
        symbol_rate=symbol_rate_gbd * 1e9,
        sps=sps,
        laser_type=laser_type,
        extinction_ratio_db=extinction_ratio_db,
        chirp_alpha=chirp_alpha,
        length_km=length_km,
        alpha_db_km=alpha_db_km,
        wavelength_nm=wavelength_nm,
        dispersion_ps_per_nm_km=dispersion_ps,
        connector_loss_db=connector_loss_db,
        splitter_ratio=splitter_ratio,
        tx_power_dbm=tx_power_dbm,
        receiver_type=receiver_type,
        apd_gain=apd_gain,
        apd_excess_exponent=0.5,
        dark_current_na=10.0,
        thermal_noise_pa_sqrt_hz=15.0e-12,
        rx_bw_ghz=rx_bw_ghz,
        equalizer_type=equalizer_type,
        equalizer_taps=equalizer_taps,
        n_symbols=n_symbols,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def _run_cached(cfg: ImddConfig) -> ImddResult:
    return run_imdd(cfg, quiet=True)


@st.cache_data(show_spinner=False)
def _sensitivity_cached(
    cfg: ImddConfig,
) -> tuple[list[float], dict[str, list[float]], dict[str, float]]:
    rop, curves, crossings = imdd_sensitivity(cfg, target_ber=1e-3)
    return (
        [float(v) for v in rop],
        {k: [float(b) for b in v] for k, v in curves.items()},
        {k: float(v) for k, v in crossings.items()},
    )


def _sidebar() -> ImddConfig:
    st.sidebar.title("IM/DD link configuration")
    preset_key = st.sidebar.selectbox("Quick preset", ["Custom", *_PRESETS.keys()], index=1)
    custom = preset_key == "Custom"
    preset = _PRESETS.get(preset_key, {})

    modulation = st.sidebar.selectbox(
        "Modulation",
        ["NRZ", "PAM4", "PAM8"],
        index=["NRZ", "PAM4", "PAM8"].index(preset.get("modulation", "PAM4")),
        disabled=not custom,
    )
    symbol_rate = st.sidebar.number_input(
        "Symbol rate (GBd)",
        min_value=1.0,
        max_value=112.0,
        value=float(preset.get("symbol_rate", 26.56)),
        step=1.0,
        disabled=not custom,
    )
    sps = st.sidebar.selectbox("Samples per symbol", [4, 8, 16], index=1)
    laser_type = st.sidebar.selectbox(
        "Laser / modulator",
        ["DML", "EML"],
        index=["DML", "EML"].index(preset.get("laser", "EML")),
        disabled=not custom,
    )
    extinction_db = st.sidebar.slider("Extinction ratio (dB)", 3.0, 20.0, 8.0, step=1.0)
    chirp_alpha = st.sidebar.slider(
        "Chirp factor (alpha)",
        0.0,
        6.0,
        0.0,
        step=0.5,
        disabled=(laser_type == "EML"),
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Channel")
    length_km = st.sidebar.slider(
        "Link length (km)",
        0.0,
        40.0,
        float(preset.get("length", 20.0)),
        step=1.0,
        disabled=not custom,
    )
    alpha_db_km = st.sidebar.slider("Fibre loss (dB/km)", 0.0, 0.6, 0.2, step=0.05)
    band = st.sidebar.selectbox(
        "Operating band",
        _BANDS,
        index=_BANDS.index(preset.get("band", "O-band (1310 nm)")),
        disabled=not custom,
    )
    is_oband = band.startswith("O")
    wavelength_nm = 1310.0 if is_oband else 1550.0
    dispersion_ps = (
        0.0 if is_oband else st.sidebar.slider("Dispersion (ps/nm/km)", 0.0, 17.0, 17.0, step=0.5)
    )
    st.sidebar.caption(
        "O-band (1310 nm) SMF has near-zero chromatic dispersion (D ≈ 0), so "
        "the C-band dispersion slider is bypassed there."
    )
    connector_loss = st.sidebar.slider("Connector losses (dB)", 0.0, 3.0, 1.0, step=0.1)
    splitter_ratio = st.sidebar.select_slider(
        "Passive splitter ratio (PON)",
        options=[1, 4, 16, 32, 64],
        value=1,
        format_func=lambda v: "none" if v == 1 else f"1:{v}",
    )
    tx_power = st.sidebar.slider("TX power (dBm)", -5.0, 10.0, 3.0, step=1.0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Receiver")
    receiver_type = st.sidebar.selectbox("Photodetector", _RECEIVERS, index=0)
    apd_gain = st.sidebar.slider(
        "APD gain (M)", 2.0, 20.0, 10.0, step=1.0, disabled=(receiver_type != "APD")
    )
    rx_bw = st.sidebar.slider("RX bandwidth (GHz)", 5.0, 60.0, 20.0, step=1.0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("DSP")
    equalizer_type = st.sidebar.selectbox("Equalizer", ["None", "FFE", "DFE"], index=1)
    eq_taps = st.sidebar.slider(
        "Equalizer taps", 3, 31, 9, step=2, disabled=(equalizer_type == "None")
    )
    n_log2 = st.sidebar.slider("Symbols (2^n)", 10, 15, 13, step=1)
    seed = st.sidebar.number_input("Seed", value=1234, step=1)

    return _build_config(
        modulation=modulation,
        symbol_rate_gbd=float(symbol_rate),
        sps=int(sps),
        laser_type=laser_type,
        extinction_ratio_db=float(extinction_db),
        chirp_alpha=float(chirp_alpha),
        length_km=float(length_km),
        alpha_db_km=float(alpha_db_km),
        wavelength_nm=float(wavelength_nm),
        dispersion_ps=float(dispersion_ps),
        connector_loss_db=float(connector_loss),
        splitter_ratio=int(splitter_ratio),
        tx_power_dbm=float(tx_power),
        receiver_type=receiver_type,
        apd_gain=float(apd_gain),
        rx_bw_ghz=float(rx_bw),
        equalizer_type=equalizer_type,
        equalizer_taps=int(eq_taps),
        n_symbols=2**n_log2,
        seed=int(seed),
    )


def _kpi_row(res: ImddResult) -> None:
    cfg = res.config
    eye = res.eye_opening
    margin = res.budget[-1][2] - res.budget[-2][2]  # ROP - sensitivity
    eop = eye["eop_db"]
    eop_txt = f"{eop:.2f} dB" + (" (clamped)" if eye.get("eop_clamped") else "")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Received power (ROP)", f"{res.rop_dbm:.2f} dBm")
    col2.metric("BER", f"{res.ber:.2e}")
    col3.metric("Errors / bits", f"{res.n_errors}/{res.n_bits}")
    col4.metric("Min. eye opening", f"{eye['min_opening']:.3f}")
    col5.metric("EOP/TDECQ", eop_txt)
    st.caption(
        f"{cfg.modulation} @ {cfg.symbol_rate / 1e9:.2f} GBd at "
        f"{cfg.wavelength_nm:.0f} nm, {cfg.laser_type} (ER {cfg.extinction_ratio_db:.0f} dB), "
        f"{cfg.length_km:.0f} km, splitter 1:{cfg.splitter_ratio}, "
        f"{cfg.receiver_type} receiver, {cfg.equalizer_type} "
        f"({cfg.equalizer_taps} taps) · link margin {margin:.1f} dB"
    )


def main() -> None:
    """Streamlit entry point for the direct-detection page."""
    st.set_page_config(page_title="Direct Detection & PON", layout="wide")
    st.title("Direct Detection & Short-Reach (PON)")
    st.caption(
        "Intensity-modulated links (NRZ / PAM4 / PAM8) with DML/EML chirp and "
        "extinction ratio, fibre + splitter losses, PIN/APD receivers and "
        "baud-spaced FFE/DFE equalization."
    )

    cfg = _sidebar()

    if st.sidebar.button("Run simulation", type="primary"):
        with st.spinner("Simulating IM/DD link..."):
            result = _run_cached(cfg)
        st.session_state["imdd_result"] = result

    if "imdd_result" not in st.session_state:
        st.info("Configure the link on the sidebar and press **Run simulation**.")
        return

    res = st.session_state["imdd_result"]
    if res.config != cfg:
        st.warning("Sidebar settings changed since the last run - press Run again.")

    _kpi_row(res)

    st.header("Eye diagrams (pre- vs post-equalizer)")
    col_pre, col_post = st.columns(2)
    with col_pre:
        st.plotly_chart(
            plot_imdd_eye(
                res.eye,
                cfg.sps,
                title="Received Eye (Pre-Equalizer)",
                thresholds=res.eye_opening["thresholds"],
                eye_metrics=res.eye_opening,
            ),
            use_container_width=True,
        )
    with col_post:
        st.plotly_chart(
            plot_imdd_eye(
                res.eye_eq,
                cfg.sps,
                title=f"Equalized Eye (Post {cfg.equalizer_type})",
                thresholds=res.eye_opening_eq["thresholds"],
                eye_metrics=res.eye_opening_eq,
            ),
            use_container_width=True,
        )
    st.caption(
        "Photocurrent at the decision instant, mean-normalised, sliced into "
        "2-symbol windows. The dashed red lines are the decision thresholds "
        "(midpoints between the measured rails). The post-equalizer eye shows "
        "what the FFE/DFE cleaned up: wider openings, tighter rails and a "
        "lower EOP/TDECQ. EOP/TDECQ is capped at 10.0 dB (IEEE-style) so a "
        "fully closed eye displays 10.0 dB rather than a huge number."
    )

    st.header("Receiver sensitivity (BER vs received power)")
    rop_grid, curves, crossings = _sensitivity_cached(cfg)
    st.plotly_chart(
        plot_sensitivity_waterfall(
            rop_grid,
            curves,
            crossings,
            target_ber=1e-3,
            title=f"Sensitivity waterfall ({cfg.modulation})",
        ),
        use_container_width=True,
    )
    st.caption(
        "Back-to-back sweep (no fibre or splitter losses) with the received "
        "power swept directly, so the curves isolate the receiver noise. BER "
        "is estimated from the equalized decision-instant statistics with a "
        "Gaussian noise-margin model, so it falls cleanly below 1e-6 at high "
        "power instead of flooring at the simulation's bit-count limit. The "
        "APD's internal gain multiplies the photocurrent before the (dominant) "
        "thermal noise, so it crosses the target BER at a lower optical power "
        "than the PIN - at the price of multiplied shot noise."
    )

    st.header("Link budget")
    st.plotly_chart(plot_link_budget_bar(res.budget), use_container_width=True)
    st.caption(
        "Waterfall of the passive optical budget: the transmitter launch power "
        "(bar 1) minus the connector, fibre and splitter losses (bars 2-4) "
        "gives the received optical power ROP (bar 5). The receiver "
        "sensitivity (bar 6) is the power at which the equalized link reaches "
        "BER 1e-3, and the system margin (bar 7) is ROP - sensitivity - drawn "
        "green when positive (the link closes with margin to spare) and red "
        "when negative."
    )


if __name__ == "__main__":
    main()
