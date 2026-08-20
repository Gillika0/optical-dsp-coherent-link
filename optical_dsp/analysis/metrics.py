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


def q_factor_from_ber(ber: float, cap_db: float | None = None) -> float:
    """Q-factor (dB) derived from the BER via the inverse error tail.

    .. math:: Q = \\sqrt 2\\,\\mathrm{erfc}^{-1}(2\\,P_b)

    ``cap_db`` clamps the result (useful for display: a zero-error run has no
    finite Q, only a lower bound).
    """
    if ber >= 0.5:
        return 0.0
    if ber <= 0.0:
        raw = 20.0 * np.log10(1e30)
        return float(min(raw, cap_db) if cap_db is not None else raw)
    raw = float(20.0 * np.log10(np.sqrt(2.0) * erfcinv(2.0 * ber)))
    return float(min(raw, cap_db) if cap_db is not None else raw)


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
    """Exact-AWGN approximate bit-error rate for Gray-coded M-QAM.

    Square M-QAM uses the classic product-of-PAM approximation:

    .. math:: P_b \\approx \\frac{4}{\\log_2 M}\\Big(1-\\frac{1}{\\sqrt M}\\Big)
              Q\\!\\left(\\sqrt{\\frac{3 E_s/N_0}{M-1}}\\right)

    The circular star 8-QAM (4 inner diagonal + 4 outer axis-aligned points)
    is sized with ``r2/r1 = (sqrt2 + sqrt6)/2`` so the inner-inner and
    inner-outer nearest-neighbour distances are equal:
    ``d_min = sqrt(2) * r1`` with ``r1 = sqrt(2/(1+(r2/r1)^2))``. The nearest
    neighbour union bound gives ``SER ~= 12 Q(d_min/sqrt(2 N0))``
    (12 nearest neighbours), ``P_b = SER / log2(8)``.
    """
    assert order in (4, 8, 16, 64, 256)
    m = float(order)
    es_no = 10.0 ** (snr_db / 10.0)
    if order == 8:
        t = (np.sqrt(2.0) + np.sqrt(6.0)) / 2.0
        r1 = np.sqrt(2.0 / (1.0 + t * t))
        dmin2 = 2.0 * r1 * r1
        q = float(q_function(np.sqrt(es_no * dmin2 / 2.0)))
        ser = 1.0 - (1.0 - q) ** 12  # complement form of the 12-neighbour bound
        p = ser / 3.0
    else:
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


@dataclass(frozen=True)
class BranchLevels:
    """Measured ADC input levels of one quadrature branch (I or Q).

    Levels are the centroids of the positive/negative sample clusters taken
    at the best sampling instant (max symbol variance over the ``sps``
    offsets). ``vpp`` is the peak-to-peak swing between them, ``spacing`` the
    adjacent-level step, ``bias`` the DC offset of the pair, ``eye_opening_db``
    the level separation relative to the pooled in-cluster noise, and
    ``clip_fraction`` the share of samples beyond the ADC full scale
    (``clip_sigma`` x RMS).
    """

    vpp: float
    level_pos: float
    level_neg: float
    bias: float
    spacing: float
    clip_fraction: float
    eye_opening_db: float
    sampling_offset: int
    n_sampled: int


def _cluster_levels(values: NDArray[np.float64]) -> tuple[float, float, float]:
    """Centroids and pooled noise of the positive/negative sample clusters."""
    pos = values[values > 0.0]
    neg = values[values < 0.0]
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan"), float("nan")
    lp = float(np.mean(pos))
    ln = float(np.mean(neg))
    noise = float(np.sqrt(0.5 * (np.var(pos) + np.var(neg))))
    return lp, ln, noise


def adc_target_metrics(
    signal: NDArray[np.complex128],
    sps: int,
    clip_sigma: float = 3.5,
) -> tuple[BranchLevels, BranchLevels]:
    """Quantify the ADC input against the receiver level specification.

    Returns ``(I, Q)`` level measurements of the signal handed to the ADC
    (photodiode + TIA output, before any DSP). For QPSK each branch should
    show two levels at ``+-Vpp/2``: equal ``spacing`` on both branches,
    near-zero ``bias`` (symmetric swing) and ``clip_fraction`` at zero.
    """
    n = len(signal) // sps * sps
    if n < 2 * sps:
        nan = float("nan")
        empty = BranchLevels(nan, nan, nan, nan, nan, nan, nan, 0, 0)
        return empty, empty
    s = signal[:n].reshape(-1, sps)
    off = int(np.argmax(np.var(s.real, axis=0) + np.var(s.imag, axis=0)))
    out: list[BranchLevels] = []
    for values in (s[:, off].real, s[:, off].imag):
        lp, ln, noise = _cluster_levels(values)
        rms = float(np.sqrt(float(np.mean(values**2))))
        full_scale = clip_sigma * rms
        clip = float(np.mean(np.abs(values) > full_scale)) if rms > 0.0 else 0.0
        vpp = lp - ln
        bias = 0.5 * (lp + ln)
        opening: float
        if vpp > 0.0 and noise > 0.0:
            opening = float(20.0 * np.log10(vpp / (2.0 * noise)))
        else:
            opening = float("nan")
        out.append(
            BranchLevels(
                vpp=float(vpp),
                level_pos=lp,
                level_neg=ln,
                bias=float(bias),
                spacing=float(vpp),
                clip_fraction=clip,
                eye_opening_db=opening,
                sampling_offset=off,
                n_sampled=int(values.size),
            )
        )
    i_res, q_res = out
    return i_res, q_res


def resolve_fec(name: str | None) -> FecCode | None:
    """Map a FEC-mode string (``none``/``hd``/``strong``) to its code."""
    return {
        None: None,
        "none": None,
        "hd": HD_FEC_RS255_239,
        "strong": STRONG_FEC_RS255_213,
    }.get(name.strip().lower() if isinstance(name, str) else name)


def line_rate_gbps(symbol_rate: float, order: int, npol: int = 2) -> float:
    """Raw line rate ``R = npol * Rs * log2(M)`` in Gbps."""
    return float(npol * symbol_rate * np.log2(order) / 1e9)


def net_rate_gbps(line_rate: float, fec_code: FecCode | None) -> float:
    """Net information rate after FEC overhead ``(1 - (n-k)/n)``."""
    if fec_code is None:
        return line_rate
    return line_rate * fec_code.k / fec_code.n


def spectral_efficiency_bits_s_hz(order: int, npol: int = 2) -> float:
    """Spectral efficiency ``npol * log2(M)`` bits/s/Hz (symbol-rate grid)."""
    return float(npol * np.log2(order))


def _snr_for_ber(order: int, ber_target: float) -> float:
    """Invert :func:`theoretical_ber_qam` by bisection (dB)."""
    lo, hi = -10.0, 45.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if theoretical_ber_qam(mid, order) > ber_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def required_osnr_db(
    symbol_rate: float,
    order: int,
    target_ber: float,
    ref_bw_hz: float,
    npol: int = 1,
) -> float:
    """OSNR (dB, in ``ref_bw_hz``) needed to reach ``target_ber`` on AWGN.

    Inverse of :func:`optical_dsp.utils.osnr_db_to_snr_db`: the per-pol symbol
    SNR is ``Es/N0 = OSNR * B_ref / (Rs * npol)``. ``npol=1`` matches the
    engine noise model (per-pol SNR = OSNR * B_ref / Rs); pass ``npol=2`` only
    for OSNR budgets defined over the *total* dual-pol power against a per-pol
    SNR target.
    """
    snr_db = _snr_for_ber(order, target_ber)
    osnr_lin = 10.0 ** (snr_db / 10.0) * symbol_rate * npol / ref_bw_hz
    return float(10.0 * np.log10(osnr_lin))


def fec_coding_gain_db(order: int, code: FecCode, target_post_ber: float = 1e-12) -> float:
    """Net coding gain (dB) of ``code`` at ``target_post_ber``.

    The gain is the SNR difference between the *uncoded* system operating at
    ``target_post_ber`` and the coded system operating at the pre-FEC BER that
    the decoder just brings down to ``target_post_ber`` (the code threshold).
    """
    p_thr = _pre_fec_threshold_ber(code, target_post_ber)
    snr_coded = _snr_for_ber(order, p_thr)
    snr_uncoded = _snr_for_ber(order, target_post_ber)
    return snr_uncoded - snr_coded


def _pre_fec_threshold_ber(code: FecCode, target_post_ber: float) -> float:
    """Pre-FEC BER where ``apply_fec`` reaches ``target_post_ber``."""
    lo, hi = 1e-6, 0.25
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if apply_fec(mid, code) > target_post_ber:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
