"""Landing page: hub linking the simulator modules.

Run with::

    streamlit run app/main.py

The dashboard is split into self-contained pages (see the sidebar or the
cards below): a coherent long-haul transmission page, a direct-detection /
short-reach (PON) page, and a theory companion.
"""

from __future__ import annotations

import optical_dsp
import streamlit as st

st.set_page_config(page_title="Optical DSP Simulator", page_icon="📡", layout="wide")

st.title("Optical DSP Simulator")
st.caption(
    "An educational suite for optical-fibre communication systems: physics-based "
    "channel models, coherent and direct-detection receiver DSP chains, and "
    "live performance metrics — all running in the browser."
)

cols = st.columns(3)
with cols[0]:
    st.markdown("### 📡 Coherent Transmission")
    st.markdown(
        "PDM QPSK / 8-QAM / 16-QAM / 64-QAM over a nonlinear SSFM fibre with "
        "a full coherent receiver DSP chain: CDC, matched filter, MIMO "
        "equalizer, FOE and blind phase search. Includes a live KPI bar "
        "(line rates, spectral efficiency, FEC coding gain, required OSNR), "
        "constellation/eye/PSD visualizations and BER-vs-OSNR waterfalls."
    )
    st.page_link(
        "pages/1_Coherent_Optical_Transmission.py",
        label="Open the coherent transmission page",
        icon="🛰️",
    )
with cols[1]:
    st.markdown("### 🔦 Direct Detection & PON")
    st.markdown(
        "Short-reach intensity-modulation links: NRZ/PAM4/PAM8 driven by a "
        "DML or EML (chirp + extinction ratio), fibre + splitter budget, "
        "PIN/APD receivers and post-detection FFE/DFE equalization. Includes "
        "multi-level eyes with EOP/linearity, sensitivity waterfalls and a "
        "full link-budget chart."
    )
    st.page_link(
        "pages/2_Direct_Detection_and_PON.py",
        label="Open the direct-detection page",
        icon="🏠",
    )
with cols[2]:
    st.markdown("### 📚 Theory & Formulations")
    st.markdown(
        "A self-contained recap of the physics and DSP behind both modules: "
        "constellation theory, the SSFM channel, OSNR/SNR conversion, blind "
        "equalization, carrier recovery, FEC coding gain and the analytical "
        "references used everywhere in the simulators."
    )
    st.page_link(
        "pages/3_Theory_and_Formulations.py",
        label="Open the theory companion",
        icon="📐",
    )

st.divider()

st.subheader("What's inside")
st.markdown(
    "- **Physics engine** — root-raised-cosine shaping, laser phase noise, "
    "split-step Fourier fibre propagation (dispersion + Kerr + loss), "
    "OSNR-controlled EDFAs, PIN/APD photodetection with shot/thermal noise."
    "\n\n"
    "- **DSP blocks** — chromatic-dispersion compensation, matched filtering, "
    "CMA/MMA MIMO equalization, frequency-offset estimation, blind phase "
    "search, and baud-spaced FFE/DFE for direct detection."
    "\n\n"
    "- **Analysis & metrics** — EVM, BER, Q-factor, FEC modelling, theoretical "
    "AWGN references, eye diagrams, waterfalls, link budgets and sensitivity "
    "curves."
)

st.caption(
    f"Engine version {optical_dsp.__version__} · simulation engine under "
    "``optical_dsp/`` · tests under ``tests/``"
)
