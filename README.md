# Coherent Optical Link & DSP Simulator

End-to-end simulation of a dual-polarisation (PDM) coherent optical transmission
link: transmit, propagate over a non-linear single-mode fibre, then recover the
payload with a full digital-signal-processing receiver chain.

Everything is pure Python + NumPy/SciPy, strictly typed, vectorised where
possible, and reproducible (seeded PRNGs everywhere).

## What it does

```
 TX                          FIBRE                       RX DSP
 ─────                       ─────                       ─────
 PRBS / random payload  ──▶  SSFM (Manakov NLSE)   ──▶  coherent 90° hybrid (LO, IQ imbalance, ADC)
 bit→symbol (Gray QAM)  │    • attenuation         │   └▶ FOE (4th-power FFT)  + CDC
 RRC pulse shaping      │    • chromatic dispersion│   └▶ matched RRC filter + symbol retiming
 PDM power split        ┘    • Kerr nonlinearity   ┘   └▶ MIMO equalizer (CMA / LS-preamble + DD-LMS)
                           EDFA (target OSNR + ASE)    └▶ blind phase search (BPS)
                                                        └▶ BER / EVM / Q-factor vs AWGN theory
```

Pipeline summary (see `optical_dsp/pipeline.py`, `LinkConfig`/`run_link`):

1. **Transmitter** — uniform random payload bits, Gray-mapped to QPSK/8-QAM/16-QAM/64-QAM
   (circular star 8-QAM: 4 inner diagonal + 4 outer axis-aligned points with
   balanced minimum distances), RRC pulse shaping (unit-energy normalisation,
   so launch power in dBm is exact), shared Wiener phase noise from the TX laser,
   PDM power split.
2. **Fibre channel** — symmetric split-step Fourier method solving the coupled
   (Manakov) nonlinear Schrödinger equation, with an analytical linear-channel
   reference for verification. A link can be split into any number of spans
   (`n_spans`): each span is propagated over `length_km / n_spans` and followed
   by an EDFA that restores the launch power, so only the accumulated nonlinear
   phase grows with the number of amplifiers while the end-of-link OSNR stays
   fixed.
3. **EDFA** — ideal-gain amplifier with additive Gaussian ASE calibrated to a
   target OSNR referenced to 0.1 nm; with multiple spans the ASE budget is
   split evenly across the amplifiers.
4. **Receiver DSP** — 4th-power frequency-offset estimation/correction,
   chromatic-dispersion compensation (full-band CD filter), matched RRC filter
   with peak-energy retiming, a 2×2 T-spaced MIMO equalizer, and block-wise
   blind phase search. For QPSK the equalizer runs a whole-frame constant-modulus
   algorithm; higher-order QAM is initialised from a least-squares estimate of
   the channel inverse over a short known preamble and then refined by
   decision-directed LMS, which converges reliably for 64-QAM (blind CMA/MMA
   gets stuck in a degenerate basin). Metrics are phase-ambiguity-resolved:
   per-polarisation BER, rotation-normalised RMS-EVM, and the implied bit
   SNR/Q-factor.
5. **FEC (optional)** — hard-decision Reed-Solomon models: the 7%-OH
   RS(255,239) "HD-FEC" and the ~20%-OH RS(255,213) "strong" code. Post-FEC BER
   is computed with a binomial bounded-distance-decoding model from the
   simulated pre-FEC BER, so the dashboard shows the coding-gain waterfall drop
   once the link operates inside the code's correction capability.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests -q              # ~105 tests: engine, DSP, metrics, end-to-end, IM/DD
ruff check .                 # lint
mypy optical_dsp             # strict type check

# dashboard (Streamlit + Plotly): landing hub + 3 pages
streamlit run app/main.py
```

Or run the engine from a console:

```bash
python -c "from optical_dsp.pipeline import LinkConfig, run_link;
r = run_link(LinkConfig(modulation='16-QAM', length_km=40, osnr_db=28, seed=42));
print(f'EVM {r.evm_percent[0]:.1f}/{r.evm_percent[1]:.1f}% BER {r.ber.ber:.1e}')"
# -> EVM 9.3/9.4% BER 9.9e-05
```

## Dashboards

Run `streamlit run app/main.py` and pick a page:

* **Coherent Optical Transmission** — PDM QPSK/8-QAM/16-QAM/64-QAM with
  per-modulation default OSNR, a BER-vs-OSNR waterfall (noise recomputed per
  point, `Es/N0 = OSNR * B_ref / Rs`) compared against AWGN theory, and a
  launch-power sweep showing the flat linear region and the Kerr-limited
  penalty above a few dBm. Constellations are drawn at a strict 1:1 aspect
  ratio with the ideal star-8QAM / square-QAM targets overlaid in red.
* **Direct Detection & PON** — NRZ/PAM4/PAM8 with DML/EML chirp and extinction
  ratio, fibre + splitter losses, PIN/APD receivers and baud-spaced FFE/DFE.
  The default O-band PAM4 100G preset (26.56 GBd, 1310 nm, 10 km) opens a clean
  eye; switching to the C-band at high baud rate and distance flags chromatic
  dispersion power fading. Eye-diagram decision thresholds are computed
  dynamically from the measured PAM cluster amplitudes, and the link-budget
  waterfall draws the transmitter power as a solid bar rising from 0 up to the
  launch level.

## Package layout

```
optical_dsp/
  physics/     transmitter (RRC, PRBS), laser, fibre channel (SSFM), EDFA
  dsp/         front-end (IQ, ADC, retiming), CDC, MIMO equalizer, carrier recovery
  analysis/    metrics (EVM/BER/Q/SNR, KPI helpers), Plotly visualisations
  imdd.py      IM/DD & short-reach (PON) engine (PAM-N, chirp, PIN/APD, FFE/DFE)
  pipeline.py  LinkConfig + run_link (coherent simulation entry point)
app/
  main.py      landing hub
  pages/       coherent transmission, direct detection & PON, theory companion
tests/         pytest suite against the library + analytical references
```

## Physics spot-check

With launch power at 0 dBm the nonlinearity is negligible, so the SSFM result
matches the analytical linear-channel solution to ~1e-6 relative error
(`tests/test_channel.py`); on a noisy link the recovered EVM sits at the
OSNR-imposed AWGN floor (e.g. QPSK at OSNR 22 dB → ~13% EVM, theoretical floor
12.7%).

## Interpreting the numbers

- **EVM** and **BER** are computed on the correctly rotated constellation
  (the CMA/BPS 4-fold phase ambiguity is resolved per polarisation).
- **OSNR** is referenced to a 0.1 nm resolution bandwidth, standard for 1550 nm
  WDM links, and converted to a per-polarisation symbol SNR via
  `Es/N0 = OSNR * B_ref / Rs` (B_ref ≈ 12.48 GHz at 1550 nm). The dashboard
  waterfall compares simulated BER against the exact AWGN formula for
  Gray-coded square M-QAM (and for the circular star 8-QAM).

## License

MIT.