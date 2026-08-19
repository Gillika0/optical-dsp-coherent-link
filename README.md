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
 PDM power split        ┘    • Kerr nonlinearity   ┘   └▶ MIMO equalizer (CMA/MMA/DD)
                           EDFA (target OSNR + ASE)    └▶ blind phase search (BPS)
                                                        └▶ BER / EVM / Q-factor vs AWGN theory
```

Pipeline summary (see `optical_dsp/pipeline.py`, `LinkConfig`/`run_link`):

1. **Transmitter** — uniform random payload bits, Gray-mapped to QPSK/16/64-QAM,
   RRC pulse shaping (unit-energy normalisation, so launch power in dBm is exact),
   shared Wiener phase noise from the TX laser, PDM power split.
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
   with peak-energy retiming, a 2×2 T-spaced MIMO equalizer (constant-modulus
   CMA for QPSK, or a bounded CMA preamble + multi-modulus MMA for higher-order
   QAM), and block-wise blind phase search. Metrics are phase-ambiguity-resolved:
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
pytest tests -q              # 68 tests: engine, DSP, metrics, end-to-end
ruff check .                 # lint
mypy optical_dsp             # strict type check

# dashboard (Streamlit + Plotly), with a "Theory" companion page
streamlit run app/main.py
```

Or run the engine from a console:

```bash
python -c "from optical_dsp.pipeline import LinkConfig, run_link;
r = run_link(LinkConfig(modulation='16-QAM', length_km=40, osnr_db=28, seed=42));
print(f'EVM {r.evm_percent[0]:.1f}/{r.evm_percent[1]:.1f}% BER {r.ber.ber:.1e}')"
# -> EVM 9.3/9.4% BER 9.9e-05
```

## Package layout

```
optical_dsp/
  physics/     transmitter (RRC, PRBS), laser, fibre channel (SSFM), EDFA
  dsp/         front-end (IQ, ADC, retiming), CDC, MIMO equalizer, carrier recovery
  analysis/    metrics (EVM/BER/Q/SNR), Plotly visualisations
  pipeline.py  LinkConfig + run_link (the one simulation entry point)
app/main.py   Streamlit dashboard
tests/        pytest suite against the library + analytical references
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
  WDM links; the dashboard waterfall compares simulated BER against the exact
  AWGN formula for Gray-coded square M-QAM.

## License

MIT.