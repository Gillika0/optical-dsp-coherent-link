"""Theory companion page for the coherent optical link & DSP simulator.

Explains what each block does and why the simulated waterfall always sits
above the ideal AWGN curve. Reachable from the sidebar navigation of the
dashboard (``streamlit run app/main.py``).
"""

from __future__ import annotations

import streamlit as st


def _md(text: str) -> None:
    st.markdown(text)


def _math(text: str) -> None:
    st.latex(text)


def _h(text: str) -> None:
    st.subheader(text)


def main() -> None:
    st.title("Coherent optical link: theory companion")
    _md(
        "This page is a short, self-contained recap of the physics and DSP "
        "behind the simulator. Every block below maps 1:1 to a module under "
        "``optical_dsp/`` and to a test under ``tests/``."
    )

    _h("1. System architecture")
    _md(
        "The link transmits **dual-polarisation (PDM)** QPSK/16-QAM/64-QAM "
        "symbols through a dispersive, nonlinear fibre, then recovers them "
        "with a full coherent receiver. The simulation is monochromatic "
        "(single optical carrier, complex baseband):"
    )
    _md(
        "```\n"
        "  PRBS -> symbols -> RRC shaping -> PDM (X/Y) -> [laser phase noise]\n"
        "      |\n"
        "      v\n"
        "  SSFM fibre (dispersion + Kerr nonlinearity + loss)\n"
        "      |\n"
        "      v\n"
        "  EDFA (ASE noise, OSNR-controlled)\n"
        "      |\n"
        "      v\n"
        "  coherent front-end: LO mix -> IQ imbalance -> ADC quantisation\n"
        "      |\n"
        "      v\n"
        "  DSP: CDC -> matched filter + retiming -> 2x2 MIMO equalizer\n"
        "       -> FOE (frequency offset) -> BPS (phase recovery)\n"
        "      |\n"
        "      v\n"
        "  metrics: EVM, BER, Q-factor (vs AWGN theory)\n"
        "```"
    )

    _h("2. Transmitter")
    _md(
        "**Symbols and bits.** The payload is drawn from a seeded uniform RNG "
        "and mapped to a Gray-coded square-QAM constellation normalized to unit "
        "mean power. Gray coding guarantees that adjacent constellation points "
        "differ by one bit, so the bit-error rate tracks the symbol-error rate "
        "without an extra factor."
    )
    _math(r"E[|s|^2] = 1")
    _md(
        "**Pulse shaping.** Each symbol is up-sampled to ``sps`` samples and "
        "convolved with a root-raised-cosine (RRC) filter of roll-off "
        "$\\beta \\in [0,1]$:"
    )
    _math(
        r"h_{\mathrm{rrc}}(t) = \frac{4\beta}{\pi\sqrt{T}}"
        r"\frac{\cos[(1+\beta)\pi t/T] + \frac{T}{4\beta t}\sin[(1-\beta)\pi t/T]}"
        r"{1 - (4\beta t/T)^2}"
    )
    _md(
        "The taps are normalised so that $\\sum_k |h_k|^2 = \\mathrm{sps}$ "
        "(unit energy per symbol), which makes the launch power set in dBm an "
        "exact optical power. Two orthogonal copies (X and Y polarisation) "
        "carry independent symbol streams, each with the full launch power."
    )
    _md(
        "**Laser phase noise.** Both the TX laser and the LO are ideal "
        "single-frequency oscillators with Lorentzian linewidth "
        "$\\Delta\\nu$. Phase noise is a continuous-time Wiener process sampled "
        "at the simulation rate, so the phase variance after a time $\\tau$ is "
        "the textbook $2\\pi\\Delta\\nu\\,\\tau$."
    )
    _math(r"\phi(t+\tau) - \phi(t) \sim \mathcal{N}\big(0,\; 2\pi\,\Delta\nu\,\tau\big)")

    _h("3. Optical channel (split-step Fourier)")
    _md(
        "The fibre is described by loss $\\alpha$, group-velocity dispersion "
        "and the Kerr nonlinearity. In the slowly-varying envelope approximation "
        "the propagation of the two polarisations follows the Manakov equation:"
    )
    _math(
        r"\frac{\partial E}{\partial z} = -\frac{\alpha}{2}E"
        r" - \mathrm{j}\frac{\beta_2}{2}\frac{\partial^2 E}{\partial t^2}"
        r" + \mathrm{j}\gamma\frac{8}{9}\|E\|^2 E"
    )
    _md(
        "The split-step Fourier method (SSFM) solves it by alternating a "
        "linear frequency-domain step (dispersion is diagonal in frequency) "
        "with a nonlinear time-domain phase step over many small slices "
        "$\\Delta z$:"
    )
    _math(
        r"E(z+\Delta z) \approx"
        r"e^{-\mathrm{j}\frac{\beta_2}{2}\omega^2\Delta z}"
        r"\,e^{-\alpha\Delta z/2}\,e^{\,\mathrm{j}\gamma\frac{8}{9}\|E\|^2\Delta z}\,E(z)"
    )
    _md(
        "With the nonlinear index set to zero the SSFM result matches the "
        "closed-form dispersion-only transfer function exactly (tested in "
        "``tests/test_channel.py``)."
    )

    _h("4. Amplifier and OSNR")
    _md(
        "The EDFA restores the launch power and adds amplified spontaneous "
        "emission (ASE). In OSNR mode the per-polarisation noise spectral "
        "density is set so that the received OSNR hits its target:"
    )
    _math(
        r"n_0 = \frac{P_s}{2\,\mathrm{OSNR}\,B_{\mathrm{ref}}},"
        r"\qquad B_{\mathrm{ref}} = \frac{c\,\Delta\lambda_{\mathrm{ref}}}{\lambda^2}"
        r" \approx 12.5\,\mathrm{GHz}"
        r"\ \mathrm{at}\ 1550\,\mathrm{nm},\ \Delta\lambda_{\mathrm{ref}}=0.1\,\mathrm{nm}"
    )
    _md(
        "Because $P_s$ is the total dual-pol power, the per-polarisation "
        "symbol SNR after a matched filter is simply"
    )
    _math(r"\frac{E_s}{N_0} = \mathrm{OSNR}\,\frac{B_{\mathrm{ref}}}{R_s}")
    _md(
        "i.e. $\\mathrm{SNR\\,[dB]} = \\mathrm{OSNR\\,[dB]} + "
        "10\\log_{10}(B_{\\mathrm{ref}}/R_s)$ with **no extra 3 dB** - the "
        "factor of two is already inside $n_0$. This is the conversion used by "
        "the waterfall on the main page."
    )
    _md(
        "**Multi-span links.** A link of length $L$ can be split into "
        "$N$ spans. Each span propagates over $L/N$ km and is followed by an "
        "EDFA that restores the launch power, so the nonlinear phase "
        "$\\phi_{\\mathrm{NL}}\\propto\\gamma P_0 L_{\\mathrm{eff}}$ accrues "
        "span by span. In OSNR mode the ASE budget is divided evenly across "
        "the amplifiers (each stage adds $n_0/N$), keeping the end-of-link "
        "OSNR at its target. Sweeping $N$ at a fixed OSNR therefore isolates "
        "the nonlinearity: the AWGN floor stays put while the Kerr penalty "
        "grows - the effect the dashboard's *Fibre spans/amplifiers* slider "
        "demonstrates."
    )

    _h("5. Coherent receiver front-end")
    _md(
        "A local oscillator at (nearly) the carrier frequency is mixed with "
        "the received field to recover the complex envelope. The model also "
        "applies gain/phase imbalance between the I and Q arms and quantises "
        "to a configurable ADC resolution, then the matched filter "
        "(identical RRC, $\\beta$ matched to the TX) compresses the signal to "
        "one sample per symbol at the optimum timing instant (selected by "
        "maximising the average symbol power)."
        "\n\nA frequency offset between TX and LO appears as a constant "
        "rotation of the constellation: $r_k = s_k e^{\\mathrm{j}2\\pi\\Delta f"
        " kT}$. **Chromatic-dispersion compensation (CDC)** inverts the linear "
        "fibre transfer function in the frequency domain, so the equalizer "
        "only has to clean residual crosstalk and gain/phase drift."
    )

    _h("6. MIMO equalizer")
    _md(
        "A T-spaced $2\\times 2$ FIR filter with 15 taps per path inverts the "
        "residual channel. It adapts blindly (no training sequence):"
    )
    _md(
        "- **CMA** (constant-modulus algorithm) is used for QPSK over the whole "
        "frame; its cost $J=\\mathbb{E}[(|y|^2-R^2)^2]$ is exactly zero at a "
        "constant-modulus solution, so it locks cleanly."
        "- For square QAM the frame starts with a bounded CMA acquisition "
        "preamble, then **MMA** (multi-modulus algorithm) takes over to the "
        "end. MMA is decision-free and robust: a pure decision-directed "
        "start-up on 16-QAM is fragile because wrong early decisions feed "
        "back into the taps and can push the filter into a degenerate basin."
    )
    _math(
        r"\mathrm{MMA:}\quad J = \mathbb{E}\big[(u^2-R_I)^2 + (v^2-R_Q)^2\big],"
        r"\qquad R_I = \frac{\mathbb{E}[\Re(s)^4]}{\mathbb{E}[\Re(s)^2]}"
    )
    _md(
        "The equalizer's causal window introduces a fixed latency of "
        "$(n_{\\mathrm{taps}}-1)/2$ symbols, which the pipeline removes before "
        "measuring EVM/BER. The adaptation error history is what the "
        '"convergence" plot shows.'
    )

    _h("7. Carrier recovery")
    _md(
        "**Frequency offset estimation (FOE).** Raising the signal to the "
        "4th power removes the modulation (all square-QAM points are invariant "
        "under 90°), leaving a tone at $4\\Delta f$ whose peak in the spectrum "
        "gives the offset. The correction is applied only when the estimate "
        "exceeds a threshold, so that spectral noise cannot trigger a false "
        "de-rotation on short frames."
    )
    _md(
        "**Blind phase search (BPS).** After frequency correction a residual "
        "Wiener phase noise remains. The frame is split into blocks and each "
        "block is rotated by all candidate phases on a grid; the rotation with "
        "the minimum distance to the constellation wins. BPS leaves a "
        "$2\\pi/M$ rotational ambiguity (square QAM is invariant under 90°), "
        "which the metrics resolve by trying all four rotations against the "
        "known transmitted data."
    )

    _h("8. Metrics")
    _md(
        "**EVM** is the RMS error-vector magnitude between the recovered and "
        "transmitted symbols, after resolving the $2\\pi/M$ ambiguity and "
        "(optionally) the optimum complex scale:"
    )
    _math(r"\mathrm{EVM_{RMS}} = 100\,\sqrt{\frac{\sum_k |r_k - s_k|^2}{\sum_k |s_k|^2}}\ \%")
    _md("**BER** counts bit errors after Gray demapping (rotation-resolved).")
    _md(
        "**Theory.** For Gray-coded square QAM in pure AWGN the bit-error "
        "probability is the standard closed form"
    )
    _math(
        r"P_b \approx \frac{4}{\log_2 M}\left(1-\frac{1}{\sqrt{M}}\right)"
        r"Q\!\left(\sqrt{\frac{3\,E_s/N_0}{M-1}}\right)"
    )
    _md(
        "with $Q(x)=\\frac12\\mathrm{erfc}(x/\\sqrt2)$. The Q-factor is the "
        "inverse-tail figure $Q = \\sqrt2\\,\\mathrm{erfc}^{-1}(2P_b)$, "
        "reported in dB."
    )

    _h("9. Forward error correction")
    _md(
        "FEC is modelled as a bounded-distance **hard-decision Reed-Solomon "
        "code over GF(256)**: each codeword of $n=255$ bytes carries $k$ "
        "information bytes and corrects any pattern of up to "
        "$t=(n-k)/2$ symbol errors. With pre-FEC bit-error probability $p$, a "
        "byte is wrong with probability $p_s = 1-(1-p)^8$, and the number of "
        "symbol errors per codeword is binomial$(n, p_s)$."
    )
    _math(
        r"p_s = 1-(1-p)^8,\qquad"
        r"X \sim \mathrm{Binomial}(n, p_s),\qquad"
        r"P_{b,\mathrm{post}} = \frac{1}{2n}"
        r"\sum_{j=t+1}^{n} j \binom{n}{j} p_s^j (1-p_s)^{n-j}"
    )
    _md(
        "When a codeword contains $j>t$ errors the decoder fails and all $j$ "
        "survive (each flips half its bits on average, hence the factor "
        "$1/2$). Below the code's threshold this sum collapses: the dashboard "
        "floors it at $10^{-15}$, which is what the green post-FEC waterfall "
        "cliff shows. Two presets are exposed: **HD-FEC** RS(255,239) at 7% "
        "overhead ($t=8$) and **Strong FEC** RS(255,213) at ~20% overhead "
        "($t=21$). Because it is a *hard-decision* model, feeding it a BER far "
        "above its threshold makes the post-FEC BER *worse* than the input - "
        "the classic behaviour of an over-loaded RS decoder."
    )

    _h("10. Why is the simulated BER above the AWGN curve?")
    _md(
        "The theory line is the **ideal limit**: no laser noise, no ADC, no "
        "filtering penalty, no blind-estimation error - only additive white "
        "Gaussian noise. The simulation includes everything the theory "
        "omits, so it can never beat it:"
    )
    _md(
        "- **Laser phase noise** (default 2×100 kHz): BPS leaves a small "
        "tracking error and the equalizer/MMA steady state adds residual "
        "phase wander."
        "- **ADC quantisation** at 8 bits costs a fraction of an LSB of EVM."
        "- **MMA steady-state residual**: a single radius cannot be exactly "
        "zero on the two-ring 16-QAM lattice, so even on a clean signal the "
        "blind equalizer sits ~4.5% EVM off the ideal points."
        "- **Residual crosstalk** between polarisations and SSFM nonlinear "
        "phase, and finite-length convergence (short frames equalize worse "
        "than infinite ones)."
        "- **DSP failure floor**: at very low OSNR the FOE/BPS phase estimate "
        "itself breaks, so the simulated BER saturates instead of following "
        "the theoretical cliff downward."
    )
    _md(
        "The gap you see at high OSNR is mostly a constant **implementation "
        "penalty of a few dB**, which the steep QAM BER cliff turns into one "
        "or two decades of BER difference. The waterfall is the right tool to "
        "see that penalty: it is *supposed* to sit above the line."
    )

    _h("11. Quick reference")
    _md(
        "| Block | Module | Default |\n"
        "|---|---|---|\n"
        "| Constellation | ``optical_dsp/utils.py`` | unit-power Gray QAM |\n"
        "| RRC shaping | ``physics/transmitter.py`` | 33 taps, $\\beta=0.2$ |\n"
        "| Laser | ``physics/laser.py`` | 100 kHz linewidth |\n"
        "| Fibre | ``physics/channel.py`` | SSFM, $\\Delta z=0.5$ km, Manakov |\n"
        "| EDFA | ``physics/amplifier.py`` | OSNR-controlled ASE |\n"
        "| Front-end | ``dsp/front_end.py`` | 8-bit ADC, peak retiming |\n"
        "| CDC | ``dsp/cdc.py`` | frequency-domain inverse |\n"
        "| Equalizer | ``dsp/equalizer.py`` | 15 taps, CMA+MMA |\n"
        "| FOE / BPS | ``dsp/carrier_recovery.py`` | 4th-power, 32 phases×16 |\n"
        "| FEC | ``analysis/metrics.py`` | RS(255,239)/RS(255,213), binomial |\n"
        "| Metrics | ``analysis/metrics.py`` | EVM, BER, Q, theory |\n"
        "| Orchestration | ``pipeline.py`` | ``LinkConfig`` + ``run_link`` |\n"
    )


if __name__ == "__main__":
    main()
