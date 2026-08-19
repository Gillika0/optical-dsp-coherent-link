"""Metrics: EVM, BER, Q-factor, FEC and theoretical AWGN references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, erfcinv
from scipy.stats import binom

from ..utils import Constellation


def q_function(x: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """Gaussian tail ``Q(x) = 0.5 * erfc(x / sqrt(2))`` (vectorised)."""
    out: NDArray[np.float64] | float = 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))
    return out


def evm_rms(
    received: NDArray[np.complex128],
    reference: NDArray[np.complex128],
    scale_optimum: bool = True,
) -> float:
    """RMS error-vector magnitude in percent.

    .. math:: EVM_\\text{RMS} = 100\\,\\sqrt{\\frac{\\sum|r_k - s_k|^2}{\\sum|s_k|^2}}

    The received symbols are optionally fitted by the optimum complex scale
    before evaluation (removes residual gain/phase only when requested).
    """
    n = min(len(received), len(reference))
    if n == 0:
        return float("nan")
    r = received[:n]
    s = reference[:n]
    if scale_optimum:
        den = float(np.vdot(r, r).real)
        alpha: complex = complex(np.vdot(s, r) / np.vdot(r, r)) if den > 0 else 1.0
        r = alpha * r
    power = float(np.sum(np.abs(s) ** 2))
    if power <= 0.0:
        return float("nan")
    err = float(np.sum(np.abs(r - s) ** 2))
    evm: float = 100.0 * float(np.sqrt(err / power))
    return evm


def resolve_rotation(
    symbols: NDArray[np.complex128],
    constellation: Constellation,
    reference_symbols: NDArray[np.complex128],
) -> int:
    """Best constellation-symmetry rotation ``k`` aligning ``symbols``.

    CMA/BPS only pin the carrier phase up to a ``2*pi/M`` ambiguity; this
    helper finds the rotation index with the smallest RMS-EVM against the
    reference. Both streams are power-normalised first so the search is
    sensitive only to phase alignment (a length/amplitude mismatch must not
    hide the winning rotation).
    """
    n = min(len(symbols), len(reference_symbols))
    if n == 0:
        return 0
    r = symbols[:n]
    ref = reference_symbols[:n]
    p_r = float(np.mean(np.abs(r) ** 2))
    p_ref = float(np.mean(np.abs(ref) ** 2))
    if p_r <= 0.0 or p_ref <= 0.0:
        return 0
    r = r / np.sqrt(p_r)
    ref = ref / np.sqrt(p_ref)
    evms = [
        evm_rms(
            r * np.exp(1j * 2.0 * np.pi * k / constellation.symmetry_order),
            ref,
            scale_optimum=False,
        )
        for k in range(constellation.symmetry_order)
    ]
    return int(min(range(constellation.symmetry_order), key=lambda k: evms[k]))


def q_factor_from_ber(ber: float) -> float:
    """Q-factor (dB) derived from the BER via the inverse error tail.

    .. math:: Q = \\sqrt 2\\,\\mathrm{erfc}^{-1}(2\\,P_b)
    """
    if ber >= 0.5 or ber <= 0.0:
        return 0.0 if ber >= 0.5 else 20.0 * np.log10(1e30)
    return float(20.0 * np.log10(np.sqrt(2.0) * erfcinv(2.0 * ber)))


@dataclass
class BerResult:
    """Bit-error-rate measurement with the resolved phase ambiguity."""

    ber: float
    n_errors: int
    n_bits: int
    best_rotation: int

    def log10(self) -> float:
        """Base-10 logarithm of the BER (``-inf`` for zero errors)."""
        return np.log10(self.ber) if self.ber > 0 else -np.inf


def measure_ber(
    received: NDArray[np.complex128],
    constellation: Constellation,
    reference_bits: NDArray[np.uint8],
    resolve_rotation: bool = True,
    symbols_per_pol: NDArray[np.complex128] | None = None,
) -> BerResult:
    """Demap and bit-compare with the transmitted reference.

    The M-fold constellation ambiguity (``2*pi/M``) introduced by BPS is
    resolved, iff ``resolve_rotation``, by trying all ``M`` rotations and
    keeping the one with the fewest bit errors.

    ``symbols_per_pol`` may shadow ``received`` for per-polarisation metrics.
    """
    sym = received if symbols_per_pol is None else symbols_per_pol
    n = min(len(sym), reference_bits.size // constellation.bits_per_symbol)
    bps = constellation.bits_per_symbol
    n_bits = n * bps
    if n_bits == 0:
        return BerResult(1.0, n_bits, n_bits, 0)

    ref = reference_bits[:n_bits]
    errs_best: int = 10**9
    rot_best: int = 0
    n_rot = constellation.symmetry_order if resolve_rotation else 1
    for k in range(n_rot):
        rotated = sym[:n] * np.exp(1j * 2.0 * np.pi * k / constellation.symmetry_order)
        idx = constellation.nearest_index(rotated)
        bits = constellation.symbols_to_bits(idx)
        errs = int(np.count_nonzero(bits != ref))
        if errs < errs_best:
            errs_best = errs
            rot_best = k
    ber = errs_best / n_bits
    return BerResult(ber, errs_best, n_bits, rot_best)


def theoretical_ber_qam(snr_db: float, order: int) -> float:
    """Exact-AWGN approximate bit-error rate for Gray-coded square M-QAM.

    .. math:: P_b \\approx \\frac{4}{\\log_2 M}\\Big(1-\\frac{1}{\\sqrt M}\\Big)
              Q\\!\\left(\\sqrt{\\frac{3 E_s/N_0}{M-1}}\\right)
    """
    assert order in (4, 16, 64, 256)
    m = float(order)
    es_no = 10.0 ** (snr_db / 10.0)
    sqrt_m = np.sqrt(m)
    p = (4.0 / np.log2(m)) * (1.0 - 1.0 / sqrt_m) * q_function(np.sqrt(3.0 * es_no / (m - 1.0)))
    return float(np.clip(p, 0.0, 1.0))


def theoretical_ber_from_evm(evm_percent: float) -> float:
    """Estimate BER from RMS-EVM using the nearest-neighbour approximation."""
    snr_db = 20.0 * np.log10(max(100.0 / evm_percent, 1e-9))
    return theoretical_ber_qam(snr_db, 4)


@dataclass(frozen=True)
class FecCode:
    """A bounded-distance hard-decision block code (Reed-Solomon over bytes).

    ``n``/``k`` are the codeword/message lengths in 8-bit symbols; the code
    corrects any pattern of up to ``t = (n - k) // 2`` symbol errors.
    """

    name: str
    n: int
    k: int

    @property
    def t(self) -> int:
        """Correction capability in symbol errors."""
        return (self.n - self.k) // 2

    @property
    def overhead(self) -> float:
        """Relative redundancy ``(n - k) / k``."""
        return (self.n - self.k) / self.k


#: 7% hard-decision FEC, the classic submarine "HD-FEC" line code.
HD_FEC_RS255_239: FecCode = FecCode("HD-FEC RS(255,239) 7%", 255, 239)

#: ~20% strong hard-decision FEC (deep overhead, low pre-FEC BER threshold).
STRONG_FEC_RS255_213: FecCode = FecCode("Strong FEC RS(255,213) 20%", 255, 213)

_FEC_BITS = 8  # RS symbols are bytes


def apply_fec(pre_fec_ber: float, code: FecCode, display_floor: float = 1e-15) -> float:
    """Post-FEC BER after bounded-distance hard-decision RS decoding.

    Independent bit errors with probability ``p`` give a byte-symbol error
    probability :math:`p_s = 1 - (1-p)^8`. The decoder corrects up to ``t``
    symbol errors per codeword; when a codeword fails, its ``j > t`` remaining
    errors survive and each flips on average half of its 8 bits. The result is
    floored at ``display_floor`` (in practice the code is error-free well
    below its threshold, and no finite simulation can measure 1e-15).
    """
    p = float(pre_fec_ber)
    if p <= 0.0:
        return 0.0
    if p >= 0.5:
        return 0.5
    psym = 1.0 - (1.0 - p) ** _FEC_BITS
    n = int(code.n)
    t = int(code.t)
    if psym <= 0.0:
        return display_floor
    j = np.arange(t + 1, n + 1, dtype=np.float64)
    pmf = binom.pmf(j, n, psym)
    exp_remaining = float(np.dot(j, pmf))
    post: float = 0.5 * exp_remaining / n
    return float(max(min(post, 0.5), display_floor))


def evm_to_snr_db(evm_percent: float) -> float:
    """SNR implied by an RMS-EVM percentage (for unit-power constellations)."""
    snr: float = 20.0 * float(np.log10(100.0 / max(evm_percent, 1e-12)))
    return snr
