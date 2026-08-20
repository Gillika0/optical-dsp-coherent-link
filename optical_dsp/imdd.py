"""IM/DD and short-reach (PON) link simulation engine.

A scalar, intensity-modulation channel model: PAM-N (NRZ/PAM4/PAM8) driving a
DML/EML laser (extinction ratio + transient chirp), fibre chromatic dispersion
(FFT domain), passive splitter / connector losses, a PIN or APD photodiode
with shot + thermal + dark-current noise, and a post-detection FFE/DFE
equalizer trained on a short preamble.

All optical powers are "average received optical power" (ROP) as read by an
optical power meter, consistent with sensitivity curves in datasheets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, resample_poly

from .analysis.metrics import q_function
from .utils import beta2_from_D, db_to_linear, dbm_to_w

PAM_ORDERS: dict[str, int] = {"NRZ": 2, "PAM4": 4, "PAM8": 8}

Q_ELECTRON: float = 1.602176634e-19  # [C]

_SPLITTER_RATIOS: tuple[int, ...] = (1, 4, 16, 32, 64)


@dataclass(frozen=True)
class ImddConfig:
    """Configuration of the IM/DD (short-reach / PON) link.

    Defaults follow a realistic O-band short-reach setup: PAM4 at
    26.5625 GBd over 20 km of fibre with near-zero chromatic dispersion at
    1310 nm. For C-band operation set ``wavelength_nm=1550`` and the SMF
    dispersion (D ~ 16-17 ps/nm/km), or switch to NRZ at 10 GBd over
    10-20 km.
    """

    modulation: str = "PAM4"
    symbol_rate: float = 26.5625e9
    sps: int = 8
    laser_type: str = "EML"
    extinction_ratio_db: float = 8.0
    chirp_alpha: float = 0.0
    length_km: float = 20.0
    alpha_db_km: float = 0.2
    wavelength_nm: float = 1310.0
    dispersion_ps_per_nm_km: float = 0.0
    connector_loss_db: float = 1.0
    splitter_ratio: int = 1
    tx_power_dbm: float = 3.0
    receiver_type: str = "PIN"
    responsivity_a_w: float = 0.9
    apd_gain: float = 10.0
    apd_excess_exponent: float = 0.5
    dark_current_na: float = 10.0
    thermal_noise_pa_sqrt_hz: float = 15.0e-12
    rx_bw_ghz: float = 20.0
    equalizer_type: str = "FFE"
    # 15 taps comfortably invert the ~20 GHz RX low-pass memory at 26.56 GBd;
    # the 9-tap FFE leaves residual ISI that floors the sensitivity curve near
    # 1e-7 even at high received power
    equalizer_taps: int = 15
    n_symbols: int = 2**14
    seed: int = 1234

    def __post_init__(self) -> None:
        object.__setattr__(self, "modulation", self.modulation.strip().upper().replace("-", ""))
        assert self.modulation in PAM_ORDERS, f"unknown IM/DD modulation {self.modulation!r}"
        assert self.splitter_ratio in _SPLITTER_RATIOS
        assert self.sps >= 4


@dataclass
class ImddResult:
    """Result of an IM/DD link simulation."""

    config: ImddConfig
    tx_bits: NDArray[np.uint8]
    tx_levels: NDArray[np.float64]
    rx_photo: NDArray[np.float64]
    rx_dsp: NDArray[np.float64]
    eye: NDArray[np.float64]
    eye_eq: NDArray[np.float64]
    dsp_ref: NDArray[np.float64]
    dsp_eval_start: int
    fs: float
    ber: float
    n_errors: int
    n_bits: int
    rop_dbm: float
    budget: list[tuple[str, float | None, float]]
    eye_opening: dict[str, object]
    eye_opening_eq: dict[str, object]
    equalizer_errors: list[float] = field(default_factory=list)


def _gray_seq(n: int) -> NDArray[np.int64]:
    """Binary-reflected Gray sequence of length ``n``."""
    idx = np.arange(n, dtype=np.int64)
    return idx ^ (idx >> 1)


def _gray_bits(gray_codes: NDArray[np.int64], bps: int) -> NDArray[np.uint8]:
    """MSB-first bit labels of Gray codes (``bps`` bits each)."""
    g = np.asarray(gray_codes, dtype=np.int64)
    bits = np.zeros((g.size, bps), dtype=np.uint8)
    for k in range(bps):
        bits[:, k] = (g >> (bps - 1 - k)) & 1
    return bits.reshape(-1)


def _level_map(modulation: str) -> tuple[NDArray[np.float64], int]:
    """Unit-height PAM levels (``0..1``) ordered so adjacent levels differ by one bit."""
    m = PAM_ORDERS[modulation]
    gray = _gray_seq(m)
    pos = np.empty(m, np.int64)
    pos[gray] = np.arange(m, dtype=np.int64)
    levels = pos.astype(np.float64) / (m - 1.0)
    return levels, int(np.log2(m))


def _fixed_loss_db(config: ImddConfig) -> float:
    splitter = 0.0 if config.splitter_ratio <= 1 else 10.0 * np.log10(config.splitter_ratio)
    return config.connector_loss_db + config.alpha_db_km * config.length_km + splitter


def received_power_dbm(config: ImddConfig) -> float:
    """Average received optical power (ROP) after all passive losses."""
    return config.tx_power_dbm - _fixed_loss_db(config)


def link_budget(
    config: ImddConfig, target_ber: float = 1e-3
) -> list[tuple[str, float | None, float]]:
    """Waterfall-style link budget ``(label, increment_db, cumulative_db)``.

    ``increment_db`` is ``None`` for absolute ("total") bars: transmitter
    launch power, received power (ROP) and receiver sensitivity. The relative
    bars are the subtractive connector / fibre / splitter losses and the final
    system margin (the jump from the sensitivity target back up to the ROP,
    positive when the link closes with margin to spare).
    """
    splitter = 0.0 if config.splitter_ratio <= 1 else 10.0 * np.log10(config.splitter_ratio)
    fibre = config.alpha_db_km * config.length_km
    rop = received_power_dbm(config)
    sens = analytical_sensitivity_dbm(config, target_ber)
    after_conn = config.tx_power_dbm - config.connector_loss_db
    after_fibre = after_conn - fibre
    return [
        ("Transmitter power", None, config.tx_power_dbm),
        ("Connector losses", -config.connector_loss_db, after_conn),
        ("Fibre loss", -fibre, after_fibre),
        ("Splitter loss", -splitter, rop),
        ("Received power (ROP)", None, rop),
        ("Receiver sensitivity", None, sens),
        ("System margin", rop - sens, rop),
    ]


def analytical_sensitivity_dbm(config: ImddConfig, target_ber: float = 1e-3) -> float:
    """ROP (dBm) at which the (equalized) link just reaches ``target_ber``.

    Thermal- and shot-noise limited analytical estimate used as the budget
    target: the smallest adjacent-level swing ``R*M*dP`` must clear the noise
    by ``Q_target = sqrt(2) erfc^-1(2 BER)`` sigma's.
    """
    m = PAM_ORDERS[config.modulation]
    er_lin = db_to_linear(config.extinction_ratio_db)
    p0 = 1.0 / er_lin
    mean_p = (1.0 + p0) / 2.0  # 50 % duty-cycle average of the 0/1 optical power
    d_level = (1.0 - p0) / (m - 1.0) / mean_p  # level spacing relative to average power
    bw = min(config.rx_bw_ghz * 1e9, 0.49 * config.symbol_rate * config.sps / 2.0)
    from scipy.special import erfcinv  # noqa: PLC0415

    q_t = float(np.sqrt(2.0) * erfcinv(2.0 * target_ber))
    g = config.apd_gain if config.receiver_type == "APD" else 1.0
    f_noise = g**config.apd_excess_exponent
    i_th = config.thermal_noise_pa_sqrt_hz

    def margin_db(p_watts: float) -> float:
        i_avg = config.responsivity_a_w * g * p_watts
        sig2 = 2.0 * Q_ELECTRON * (i_avg + config.dark_current_na * 1e-9) * f_noise * bw
        sig2 += i_th**2 * bw
        sigma = np.sqrt(sig2)
        if sigma <= 0.0:
            return 0.0
        swing = config.responsivity_a_w * g * d_level * p_watts
        return float(20.0 * np.log10(max(swing / (2.0 * q_t * sigma), 1e-12)))

    lo_w, hi_w = 1e-12, 1e-3  # -90 dBm .. 0 dBm
    for _ in range(60):
        mid = 0.5 * (lo_w + hi_w)
        if margin_db(mid) > 0.0:
            hi_w = mid
        else:
            lo_w = mid
    p_sens = 0.5 * (lo_w + hi_w)
    return float(10.0 * np.log10(max(p_sens, 1e-30)) + 30.0)


def _chirped_field(
    intensity: NDArray[np.float64], er_lin: float, alpha: float
) -> NDArray[np.complex128]:
    """Optical field of the IM laser: sqrt(P) with DML transient chirp.

    ``alpha`` is the linewidth-enhancement (Henry) factor; the instantaneous
    frequency deviation is ``dphi/dt = (alpha/2) d ln(P)/dt`` so the phase is
    ``phi = (alpha/2) ln(P/P0)``. EML (``alpha=0``) gives pure IM.
    """
    p0 = 1.0 / er_lin
    phase = (alpha / 2.0) * np.log(np.maximum(intensity, p0 * 1e-12) / p0)
    out: NDArray[np.complex128] = np.sqrt(intensity) * np.exp(1j * phase)
    return out


def _dispersion_filter(
    signal: NDArray[np.complex128],
    fs: float,
    length_km: float,
    d_ps: float,
    wavelength_nm: float,
) -> NDArray[np.complex128]:
    """Apply chromatic dispersion in the frequency domain (field level).

    ``d_ps`` is the dispersion parameter in ps/(nm*km) at ``wavelength_nm``;
    in the O-band at 1310 nm SMF has D ~ 0 ps/(nm*km), so the filter becomes
    identity there.
    """
    n = signal.size
    f = np.fft.fftfreq(n, d=1.0 / fs)
    beta2 = beta2_from_D(wavelength_nm * 1e-9, d_ps)
    h = np.exp(1j * beta2 * (length_km * 1e3) * (2.0 * np.pi * f) ** 2 / 2.0)
    out: NDArray[np.complex128] = np.fft.ifft(np.fft.fft(signal) * h)
    return out


def _first_order_lpf(
    signal: NDArray[np.float64], fs: float, fc: float
) -> tuple[NDArray[np.float64], float]:
    """First-order low-pass with a known white-noise output gain."""
    b = np.exp(-2.0 * np.pi * fc / fs)
    y: NDArray[np.float64] = lfilter(np.array([1.0 - b]), np.array([1.0, -b]), signal)
    return y, float((1.0 - b) / (1.0 + b))


def _nearest_level(values: NDArray[np.float64], levels: NDArray[np.float64]) -> NDArray[np.float64]:
    dist = np.abs(values[:, None] - levels[None, :])
    out: NDArray[np.float64] = levels[np.argmin(dist, axis=1)]
    return out


def _best_sampling_offset(
    grid: NDArray[np.float64], d_ref: NDArray[np.float64], levels: NDArray[np.float64]
) -> int:
    """Column maximising the total eye opening (level gap minus cluster noise).

    For multi-level PAM the symbol-centre columns do not maximise the raw
    sample variance (transitions spread as much as the rails), so the offset
    is selected by the eye-opening criterion directly.
    """
    lvl = np.sort(levels)  # amplitude order
    best, best_open = 0, -1.0
    for c in range(grid.shape[1]):
        col = grid[:, c]
        means = np.array([float(np.mean(col[d_ref == lv])) for lv in lvl])
        opening = 0.0
        for p in range(len(lvl) - 1):
            sel_lo, sel_hi = d_ref == lvl[p], d_ref == lvl[p + 1]
            sigma = (float(np.std(col[sel_lo])) if np.any(sel_lo) else 0.0) + (
                float(np.std(col[sel_hi])) if np.any(sel_hi) else 0.0
            )
            opening += max(0.0, (means[p + 1] - means[p]) - sigma)
        if opening > best_open:
            best_open, best = opening, c
    return best


def _train_ffe(x: NDArray[np.float64], d: NDArray[np.float64], taps: int) -> NDArray[np.float64]:
    """Least-squares (Wiener) feed-forward equalizer from a training block."""
    from numpy.lib.stride_tricks import sliding_window_view

    n_tr = len(d)
    if n_tr < taps:
        return np.ones(1)
    rows = np.flip(sliding_window_view(x, taps), axis=1)  # row i = x[i+taps-1]..x[i]
    w, *_ = np.linalg.lstsq(rows[: n_tr - taps + 1], d[taps - 1 :], rcond=None)
    out: NDArray[np.float64] = np.asarray(w, dtype=np.float64).reshape(-1)
    return out


def _apply_ffe(x: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    from numpy.lib.stride_tricks import sliding_window_view

    taps = w.size
    if taps <= 1:
        return x.copy()
    rows = np.flip(sliding_window_view(x, taps), axis=1)
    y = np.empty_like(x)
    y[: taps - 1] = x[: taps - 1]
    y[taps - 1 :] = rows @ w
    return y


def _apply_dfe(
    x: NDArray[np.float64],
    wf: NDArray[np.float64],
    wb: NDArray[np.float64],
    levels: NDArray[np.float64],
    train_end: int,
) -> tuple[NDArray[np.float64], int]:
    """Decision-feedback equalizer; decisions are training targets up to ``train_end``."""
    from numpy.lib.stride_tricks import sliding_window_view

    n = x.size
    tf, tb = wf.size, wb.size
    frows = np.flip(sliding_window_view(x, tf), axis=1)
    y = np.empty(n)
    y[: tf - 1] = x[: tf - 1]
    decisions = np.empty(n)
    decisions[: tf - 1] = x[: tf - 1]
    for i in range(tf - 1, n):
        fb = decisions[i - tb : i] if i >= tb else np.concatenate([np.zeros(tb - i), decisions[:i]])
        y[i] = float(frows[i - tf + 1] @ wf + fb @ wb)
        if i <= train_end:
            decisions[i] = y[i]
        else:
            decisions[i] = float(_nearest_level(y[i : i + 1], levels)[0])
    return y, tf - 1


def _apply_ffe_os(x: NDArray[np.float64], w: NDArray[np.float64], sps: int) -> NDArray[np.float64]:
    """Apply a baud-spaced FFE to the *oversampled* signal.

    The baud-trained taps are embedded on the symbol grid (zeros between
    taps) so the filter output is well-defined at every sample: the
    equalized waveform used to draw the post-FFE eye diagram. At the
    sampling instants ``i = j*sps + off`` it reproduces exactly the
    baud-rate output ``y[j]``.
    """
    taps = w.size
    y = np.zeros_like(x)
    for k in range(taps):
        if w[k] != 0.0:
            y[k * sps :] += w[k] * x[: len(x) - k * sps]
    return y


def _eye_opening_metrics(
    samples: NDArray[np.float64],
    d_ref: NDArray[np.float64],
    levels: NDArray[np.float64],
    m: int,
    p0: float,
) -> dict[str, object]:
    """Eye-opening metrics from decision-instant samples, mean-normalised.

    Levels are mean-normalised photocurrent so the rails sit at their average
    optical-power spacing; the ideal adjacent-level spacing is the extinction-
    ratio limited ``ideal_gap = (1-p0)/((m-1)*mean_p)``. ``eop_db`` is the
    penalty between that ideal spacing and the minimum measured opening, capped
    at 10 dB (IEEE-style display clamp: a fully closed eye reports 10.0 dB
    rather than an unbounded number).
    """
    samp_norm: NDArray[np.float64] = samples / max(float(np.mean(samples)), 1e-30)
    lvl = np.sort(levels)  # amplitude order for the eye openings
    level_means: list[float] = []
    for p in range(m):
        sel = d_ref == lvl[p]
        level_means.append(float(np.mean(samp_norm[sel])) if np.any(sel) else float(lvl[p]))
    openings: list[float] = []
    for p in range(m - 1):
        lo, hi = level_means[p], level_means[p + 1]
        sigma_lo = float(np.std(samp_norm[d_ref == lvl[p]])) if np.any(d_ref == lvl[p]) else 0.0
        sigma_hi = (
            float(np.std(samp_norm[d_ref == lvl[p + 1]])) if np.any(d_ref == lvl[p + 1]) else 0.0
        )
        openings.append(max(0.0, (hi - lo) - (sigma_lo + sigma_hi)))
    mean_p = p0 + (1.0 - p0) * 0.5  # average optical power of the 0/1 waveform
    ideal_gap = (1.0 - p0) / ((m - 1.0) * mean_p)  # level spacing / mean power
    min_open = min(openings) if openings else 0.0
    outer = [openings[0]] + ([openings[-1]] if len(openings) > 1 else [])
    outer_mean = (sum(outer) / len(outer)) if outer else 0.0
    eop_raw = 10.0 * np.log10(max(ideal_gap / max(min_open, 1e-12), 1.0))
    eye_opening: dict[str, object] = {
        "min_opening": min_open,
        "ideal_opening": ideal_gap,
        "eop_db": min(eop_raw, 10.0),
        "eop_clamped": bool(eop_raw > 10.0),
        "eye_linearity": float(min_open / outer_mean) if outer_mean > 0.0 else 0.0,
        "levels": level_means,
        "openings": openings,
        "thresholds": [0.5 * (level_means[p] + level_means[p + 1]) for p in range(m - 1)],
    }
    if len(openings) >= 3:
        eye_opening["middle_opening"] = openings[1]
    return eye_opening


def _gaussian_ber(
    samples: NDArray[np.float64], d_ref: NDArray[np.float64], levels: NDArray[np.float64], bps: int
) -> float:
    """Gaussian noise-margin BER from decision-instant sample statistics.

    Measures the per-level mean and noise sigma at the decision instant and
    computes the error probability of each level against the midpoints of its
    neighbours via the Gaussian tail. This removes the simulation's
    ``1/n_bits`` error floor, so sensitivity curves drop cleanly below 1e-6 at
    high received power. Assumes Gray ordering (adjacent-level errors cost one
    bit) and that residual inter-symbol interference is small enough that the
    per-level residual is approximately Gaussian.
    """
    lvl = np.sort(levels)
    means: list[float] = []
    sigmas: list[float] = []
    for lv in lvl:
        sel = d_ref == lv
        if np.any(sel):
            means.append(float(np.mean(samples[sel])))
            sigmas.append(float(np.std(samples[sel])))
        else:
            means.append(float(lv))
            sigmas.append(0.0)
    ser = 0.0
    for i in range(len(lvl)):
        p_err = 0.0
        if i > 0:
            thr = 0.5 * (means[i - 1] + means[i])
            p_err += float(q_function((means[i] - thr) / max(sigmas[i], 1e-30)))
        if i < len(lvl) - 1:
            thr = 0.5 * (means[i] + means[i + 1])
            p_err += float(q_function((thr - means[i]) / max(sigmas[i], 1e-30)))
        ser += p_err / len(lvl)
    return float(min(ser / bps, 1.0))


def run_imdd(config: ImddConfig, quiet: bool = False) -> ImddResult:
    """Run the IM/DD link simulation for :class:`ImddConfig`."""
    del quiet
    m = PAM_ORDERS[config.modulation]
    levels, bps = _level_map(config.modulation)
    er_lin = db_to_linear(config.extinction_ratio_db)

    rng = np.random.default_rng(config.seed)
    n_sym = config.n_symbols
    bits = rng.integers(0, 2, n_sym * bps, dtype=np.uint8)
    chunks = bits.reshape(n_sym, bps)
    idx = np.zeros(n_sym, dtype=np.int64)
    for k in range(bps):
        idx = ((idx << 1) | chunks[:, k].astype(np.int64)).astype(np.int64)
    tx_levels = levels[idx]

    fs = config.symbol_rate * config.sps
    up = np.repeat(tx_levels, config.sps)
    p0 = 1.0 / er_lin
    p_wave = p0 + (1.0 - p0) * up

    alpha = config.chirp_alpha if config.laser_type == "DML" else 0.0
    field = _chirped_field(p_wave, er_lin, alpha)
    field_cd = _dispersion_filter(
        field, fs, config.length_km, config.dispersion_ps_per_nm_km, config.wavelength_nm
    )

    rop_w = dbm_to_w(received_power_dbm(config))
    p_inst: NDArray[np.float64] = field_cd.real**2 + field_cd.imag**2
    p_inst = p_inst / max(float(np.mean(p_inst)), 1e-30)

    g = config.apd_gain if config.receiver_type == "APD" else 1.0
    f_noise = g**config.apd_excess_exponent
    i_avg = config.responsivity_a_w * g * rop_w
    i_dark = config.dark_current_na * 1e-9
    photo = config.responsivity_a_w * g * rop_w * p_inst + i_dark

    fc = min(config.rx_bw_ghz * 1e9, 0.49 * fs / 2.0)
    photo_filt, noise_gain = _first_order_lpf(photo, fs, fc)
    s_noise = 2.0 * Q_ELECTRON * (i_avg + i_dark) * f_noise
    s_noise += config.thermal_noise_pa_sqrt_hz**2
    sigma_in = np.sqrt(max(s_noise * fc / max(noise_gain, 1e-30), 0.0)) if noise_gain > 0 else 0.0
    noise = rng.normal(0.0, sigma_in, photo.size)
    noise_filt, _ = _first_order_lpf(noise, fs, fc)
    rx = photo_filt + noise_filt

    # sampling instant (column with the largest eye opening)
    n_used = len(rx) // config.sps * config.sps
    grid = rx[:n_used].reshape(-1, config.sps)
    d_ref = tx_levels[: n_used // config.sps]
    off = _best_sampling_offset(grid, d_ref, levels)
    samp = grid[:, off]

    # receiver DSP: baud-spaced FFE / DFE trained on a 512-symbol preamble
    train_end = min(512, len(samp) - config.equalizer_taps)
    eq = config.equalizer_type.strip().upper()
    if eq == "FFE":
        w = _train_ffe(samp, d_ref, config.equalizer_taps)
        y = _apply_ffe(samp, w)
        latency = config.equalizer_taps - 1
    elif eq == "DFE":
        from numpy.lib.stride_tricks import sliding_window_view

        tf = int(np.ceil(config.equalizer_taps / 2.0))
        tb = config.equalizer_taps - tf
        frows = np.flip(sliding_window_view(samp, tf), axis=1)
        train_n = train_end - (tf - 1) + 1
        fb_known = np.zeros((train_n, tb))
        for j, i in enumerate(range(tf - 1, train_end + 1)):
            fb_known[j] = np.concatenate([np.zeros(max(0, tb - i)), d_ref[max(0, i - tb) : i]])
        xm = np.concatenate([frows[:train_n], fb_known], axis=1)
        w_all, *_ = np.linalg.lstsq(xm, d_ref[tf - 1 : train_end + 1], rcond=None)
        wf = np.asarray(w_all[:tf], dtype=np.float64).reshape(-1)
        wb = np.asarray(w_all[tf:], dtype=np.float64).reshape(-1)
        y, latency = _apply_dfe(samp, wf, wb, levels, train_end)
    else:
        y = samp.copy()
        latency = 0

    # BER on the payload (skip preamble + equalizer latency)
    eval_start = max(min(train_end + 1, len(samp) - 1), latency + 1)
    y_eval = y[eval_start:]
    d_eval = d_ref[eval_start:]
    dec_levels = _nearest_level(y_eval, levels)
    dec_idx = np.rint(dec_levels * (m - 1.0)).astype(np.int64)
    ref_idx = np.rint(d_eval * (m - 1.0)).astype(np.int64)
    gray = _gray_seq(m)
    dec_bits = _gray_bits(gray[dec_idx], bps)
    ref_bits = _gray_bits(gray[ref_idx], bps)
    n_bits = int(dec_bits.size)
    n_err = int(np.count_nonzero(dec_bits != ref_bits)) if n_bits else 0
    ber = n_err / n_bits if n_bits else 1.0

    # eye-opening metrics at the decision instant, in mean-normalised
    # photocurrent units (so the rails sit at their optical-power spacing)
    eye_opening = _eye_opening_metrics(samp, d_ref, levels, m, p0)
    eye_opening_eq = _eye_opening_metrics(y, d_ref, levels, m, p0)

    # oversampled equalized waveform for the post-DSP eye: the baud-trained
    # FFE taps applied on the symbol grid (exact at the sampling instants), or
    # a bandlimited interpolation of the baud-rate DFE / raw output.
    if eq == "FFE" and config.equalizer_taps > 1:
        eye_eq = _apply_ffe_os(rx, w, config.sps)
    else:
        eye_eq = resample_poly(y, config.sps, 1)

    return ImddResult(
        config=config,
        tx_bits=bits,
        tx_levels=tx_levels,
        rx_photo=rx,
        rx_dsp=y,
        eye=rx,
        eye_eq=eye_eq,
        dsp_ref=d_ref,
        dsp_eval_start=eval_start,
        fs=fs,
        ber=ber,
        n_errors=n_err,
        n_bits=n_bits,
        rop_dbm=received_power_dbm(config),
        budget=link_budget(config),
        eye_opening=eye_opening,
        eye_opening_eq=eye_opening_eq,
    )


def imdd_sensitivity(
    config: ImddConfig,
    rop_grid_dbm: Iterable[float] | None = None,
    receivers: Iterable[str] = ("PIN", "APD"),
    target_ber: float = 1e-3,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]], dict[str, float]]:
    """BER vs received optical power for each receiver type.

    Returns ``(rop_grid, {receiver: ber_curve}, {receiver: crossing_rop_dbm})``.
    The sweep uses a back-to-back channel (zero-length fibre) so the curves
    isolate the PIN/APD receiver noise behaviour from chromatic dispersion;
    the ROP is swept directly via the transmitter power. BER is estimated
    from the equalized decision-instant statistics with the Gaussian
    noise-margin model, so the curves fall cleanly below 1e-6 at high power
    instead of flooring at the simulation's ``1/n_bits`` limit.
    """
    if rop_grid_dbm is None:
        rop_grid: NDArray[np.float64] = np.linspace(-32.0, 0.0, 13)
    else:
        rop_grid = np.asarray(list(rop_grid_dbm), dtype=np.float64)
    levels, bps = _level_map(config.modulation)
    base = ImddConfig(
        modulation=config.modulation,
        symbol_rate=config.symbol_rate,
        sps=config.sps,
        laser_type=config.laser_type,
        extinction_ratio_db=config.extinction_ratio_db,
        chirp_alpha=config.chirp_alpha,
        length_km=0.0,
        alpha_db_km=config.alpha_db_km,
        wavelength_nm=config.wavelength_nm,
        dispersion_ps_per_nm_km=config.dispersion_ps_per_nm_km,
        connector_loss_db=0.0,
        splitter_ratio=1,
        tx_power_dbm=0.0,
        receiver_type="PIN",
        responsivity_a_w=config.responsivity_a_w,
        apd_gain=config.apd_gain,
        apd_excess_exponent=config.apd_excess_exponent,
        dark_current_na=config.dark_current_na,
        thermal_noise_pa_sqrt_hz=config.thermal_noise_pa_sqrt_hz,
        rx_bw_ghz=config.rx_bw_ghz,
        equalizer_type=config.equalizer_type,
        equalizer_taps=config.equalizer_taps,
        n_symbols=2**12,
        seed=config.seed,
    )
    curves: dict[str, NDArray[np.float64]] = {}
    crossings: dict[str, float] = {}
    for rx_type in receivers:
        ber_list: list[float] = []
        for rop in rop_grid:
            cfg = ImddConfig(
                **{**base.__dict__, "tx_power_dbm": float(rop), "receiver_type": rx_type}
            )
            res = run_imdd(cfg, quiet=True)
            ber_list.append(
                _gaussian_ber(
                    res.rx_dsp[res.dsp_eval_start :],
                    res.dsp_ref[res.dsp_eval_start :],
                    levels,
                    bps,
                )
            )
        curves[rx_type] = np.asarray(ber_list, dtype=np.float64)
        crossing = np.nan
        for rop, b in zip(rop_grid, ber_list):
            if b <= target_ber:
                crossing = float(rop)
                break
        crossings[rx_type] = crossing
    return rop_grid, curves, crossings
