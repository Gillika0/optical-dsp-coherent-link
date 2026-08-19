"""Time-domain adaptive MIMO FIR equalizer: CMA, MMA and DD-LMS passes.

Models an invertible static 2x2 MIMO channel and learns its inverse with a
T-spaced 2x2 FIR filter, updated by complex stochastic gradient descent.

Update laws (exact Wirtinger gradients, so no ambiguity of sign):

* CMA cost ``J = E[(|y|^2 - R^2)^2]``

  ``w_k <- w_k + m * (R^2 - |y_k|^2) * conj(y_k) * x``

* MMA cost ``J = E[(u^2 - R1)^2 + (v^2 - R2)^2]``, ``y = u + jv``

  ``w_k <- w_k + m * (R1 - u_k^2) * u_k * x + j m * (R2 - v_k^2) * v_k * x``

The input ``x`` is the stacked tap-delay vector of both polarisations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..utils import Constellation


@dataclass
class MimoEqualizer:
    """2x2 MIMO LMS-type equalizer with CMA and MMA error functions.

    Parameters
    ----------
    n_taps:
        Filter length per MIMO path (must be odd).
    mu_cma:
        CMA step size.
    mu_mma:
        MMA/DD step size.
    leakage:
        optional tap leakage per step (``0`` = off).
    seed:
        RNG seed for tap initialisation.
    """

    n_taps: int = 15
    mu_cma: float = 1e-3
    mu_mma: float = 1e-4
    #: tap leakage per adaptation step (``0`` disables); bounds the stochastic
    #: random walk of decision-directed adaptation under a rotating carrier
    leakage: float = 0.0
    seed: int | None = 3
    normalise_input: bool = True

    #: complex weight matrix, shape (2*n_taps, 2)  [in, out]
    weights: NDArray[np.complex128] = field(init=False, repr=False)
    #: applied input gain (``1 / sqrt(mean(|x0|^2 + |x1|^2))``)
    input_gain: float = field(init=False, repr=False, default=1.0)
    _errs: list[float] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        assert self.n_taps % 2 == 1, "odd tap count required"
        rng = np.random.default_rng(self.seed)
        # Physical center-tap "identity" initialization (each output starts
        # locked to its own input); random init is pathological for CMA on
        # constant-modulus signals because every unitary mix is also CM.
        self.weights = rng.normal(0.0, 1e-6, size=(2 * self.n_taps, 2)) + 1j * rng.normal(
            0.0, 1e-6, size=(2 * self.n_taps, 2)
        )
        c = (self.n_taps - 1) // 2
        self.weights[c, 0] = 1.0 + 0j
        self.weights[self.n_taps + c, 1] = 1.0 + 0j

    # ------------------------------------------------------------------ #
    def _stacked_inputs(
        self, x0: NDArray[np.complex128], x1: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Per-sample tap-delay vectors for both modes -> (N, 2*ntaps).

        Optionally normalises the instantaneous power to unity (implicit AGC);
        otherwise CMA/MMA starve or diverge depending on the absolute scale
        of the photocurrent.
        """
        n = len(x0)
        hist = self.n_taps - 1
        pad0 = np.concatenate([np.zeros(hist, dtype=x0.dtype), x0])
        pad1 = np.concatenate([np.zeros(hist, dtype=x1.dtype), x1])
        if self.normalise_input:
            power = float(np.mean(np.abs(x0) ** 2 + np.abs(x1) ** 2))
            gain = float(np.sqrt(1.0 / max(power, 1e-30)))
            self.input_gain = gain
            pad0 = pad0 * gain
            pad1 = pad1 * gain
        win0 = np.lib.stride_tricks.sliding_window_view(pad0, self.n_taps)
        win1 = np.lib.stride_tricks.sliding_window_view(pad1, self.n_taps)
        assert win0.shape == (n, self.n_taps) and win1.shape == (n, self.n_taps)
        return np.concatenate([win0, win1], axis=1)

    def _adapt_block(
        self,
        xall: NDArray[np.complex128],
        n_iter: int | None,
        constellation: Constellation | None,
        mode: str,
        mu: float,
        start: int,
        err_history: list[float],
        err_every: int = 64,
    ) -> NDArray[np.complex128]:
        """One adaptation pass over ``xall[start:start+n_iter]``.

        ``mode`` selects the error function: ``"cma"`` (constant modulus),
        ``"mma"`` (multi-modulus for square QAM) or ``"dd"`` (decision
        directed LMS, zero error at a converged solution on clean data).

        The mean absolute error is appended to ``err_history`` every
        ``err_every`` symbols (plus the final partial chunk), so the
        dashboard's convergence curve has enough points to be visible.
        Returns the filtered outputs for the **whole** frame ``(N, 2)``.
        """
        n_total = xall.shape[0]
        if n_iter is None:
            n_iter = n_total - start
        else:
            n_iter = int(min(n_iter, n_total - start))

        if mode == "cma":
            r2_i = r2_q = constellation.cma_radius2 if constellation is not None else 1.0
        elif constellation is not None:
            r2_i, r2_q = constellation.decision_radius2_i, constellation.decision_radius2_q
        else:
            r2_i = r2_q = 1.0
        dd_const = constellation if mode == "dd" and constellation is not None else None

        w = self.weights
        lk = self.leakage if mode == "dd" else 0.0
        err_acc = 0.0
        count = 0

        for n in range(start, start + n_iter):
            x = xall[n]
            y = x @ w.conj()
            if mode == "dd" and dd_const is not None:
                d_sym = dd_const.symbols
                e0 = np.conj(d_sym[dd_const.nearest_index(y[0:1])[0]] - y[0])
                e1 = np.conj(d_sym[dd_const.nearest_index(y[1:2])[0]] - y[1])
            elif mode == "mma":
                u = y.real
                v = y.imag
                e0 = u[0] * (r2_i - u[0] ** 2) + 1j * v[0] * (r2_q - v[0] ** 2)
                e1 = u[1] * (r2_i - u[1] ** 2) + 1j * v[1] * (r2_q - v[1] ** 2)
            else:
                e0 = (r2_i - np.abs(y[0]) ** 2) * np.conj(y[0])
                e1 = (r2_i - np.abs(y[1]) ** 2) * np.conj(y[1])
            err_acc += abs(e0) + abs(e1)
            count += 1
            if lk > 0.0:
                w *= 1.0 - lk
            w[:, 0] += mu * x * e0
            w[:, 1] += mu * x * e1
            if count % err_every == 0:
                err_history.append(err_acc / (2.0 * err_every))
                err_acc = 0.0

        rem = count % err_every
        if rem:
            err_history.append(err_acc / (2.0 * rem))

        # Re-filter the *whole* frame with the final taps: the returned traces
        # must never contain the zero-padded regions of the adaptation window.
        out = xall @ w.conj()
        return out

    # ------------------------------------------------------------------ #
    def run_cma(
        self,
        x0: NDArray[np.complex128],
        x1: NDArray[np.complex128],
        constellation: Constellation | None = None,
        n_iter: int | None = 4000,
        start: int = 0,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], list[float]]:
        """Pre-converge with CMA; returns ``(y0, y1, error_history)``.

        ``constellation`` supplies the CMA dispersion constant when given
        (defaults to radius-squared 1 otherwise).
        """
        xall = self._stacked_inputs(x0, x1)
        history: list[float] = []
        out = self._adapt_block(xall, n_iter, constellation, "cma", self.mu_cma, start, history)
        y0, y1 = out[:, 0].copy(), out[:, 1].copy()
        return y0, y1, history

    def run_mma(
        self,
        x0: NDArray[np.complex128],
        x1: NDArray[np.complex128],
        constellation: Constellation,
        n_iter: int | None = None,
        start: int = 0,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], list[float]]:
        """Adapt with MMA (steady state) and filter the whole frame."""
        xall = self._stacked_inputs(x0, x1)
        history: list[float] = []
        out = self._adapt_block(xall, n_iter, constellation, "mma", self.mu_mma, start, history)
        y0, y1 = out[:, 0].copy(), out[:, 1].copy()
        return y0, y1, history

    def run_dd(
        self,
        x0: NDArray[np.complex128],
        x1: NDArray[np.complex128],
        constellation: Constellation,
        n_iter: int | None = None,
        start: int = 0,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], list[float]]:
        """Decision-directed LMS steady state (zero error at a clean solution)."""
        xall = self._stacked_inputs(x0, x1)
        history: list[float] = []
        out = self._adapt_block(xall, n_iter, constellation, "dd", self.mu_mma, start, history)
        y0, y1 = out[:, 0].copy(), out[:, 1].copy()
        return y0, y1, history

    def equalize(
        self,
        x0: NDArray[np.complex128],
        x1: NDArray[np.complex128],
        constellation: Constellation,
        n_cma: int | None = 4000,
        n_mma: int = 1000,
        n_dd: int | None = None,
        mode: str = "auto",
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], list[float]]:
        """Equalize a symbol-synchronised 2x2 stream.

        The front end (CDC + matched filter) already removes most of the
        channel memory, so this block mainly cleans residual crosstalk and
        gain/phase drift:

        * ``mode="auto"`` (default): QPSK -> whole-frame CMA (its error is
          exactly zero at a constant-modulus solution, so it locks cleanly);
          higher-order QAM -> a bounded CMA acquisition preamble (``n_cma``,
          capped at ``cma_preamble`` symbols) followed by MMA for the rest of
          the frame.  A pure DD start-up on squared QAM is fragile: wrong
          decisions feed back into the taps and the filter can drift into a
          degenerate basin (seed- and frame-length dependent).  The blind
          CMA+MMA chain never exhibits that failure, at the small cost of a
          slightly higher steady-state EVM.
        * ``mode="cma+mma"``: classical blind preamble - CMA for ``n_cma``,
          MMA for ``n_mma`` symbols, then DD to the end.
        """
        self._errs = []
        xall = self._stacked_inputs(x0, x1)
        n_total = xall.shape[0]
        n_cma = n_total if n_cma is None else int(min(n_cma, n_total))
        cma_preamble = int(min(n_cma, 1200))

        order = constellation.order
        if mode == "auto":
            if order == 4:
                mode = "cma"
            else:
                self._adapt_block(
                    xall, cma_preamble, constellation, "cma", self.mu_cma, 0, self._errs
                )
                self._adapt_block(
                    xall, None, constellation, "mma", self.mu_mma, cma_preamble, self._errs
                )
                out = xall @ self.weights.conj()
                y0, y1 = out[:, 0].copy(), out[:, 1].copy()
                return y0, y1, self._errs
        if mode == "dd":
            self._adapt_block(xall, None, constellation, "dd", self.mu_mma, 0, self._errs)
        elif mode == "cma":
            self._adapt_block(xall, n_total, constellation, "cma", self.mu_cma, 0, self._errs)
        else:  # cma + mma
            self._adapt_block(xall, n_cma, constellation, "cma", self.mu_cma, 0, self._errs)
            self._adapt_block(xall, n_mma, constellation, "mma", self.mu_mma, n_cma, self._errs)
            n_dd = max(0, min(n_total - n_cma - n_mma, n_total) if n_dd is None else n_dd)
            if n_dd > 0:
                self._adapt_block(
                    xall, n_dd, constellation, "dd", self.mu_mma, n_cma + n_mma, self._errs
                )

        out = xall @ self.weights.conj()
        y0, y1 = out[:, 0].copy(), out[:, 1].copy()
        return y0, y1, self._errs

    @property
    def errors(self) -> list[float]:
        """Most recent adaptation error history."""
        return self._errs
