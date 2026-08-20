"""End-to-end PDM coherent link processing pipeline.

Wires the physics layer, DSP receiver chain and metrics into one function so
the dashboard and tests share exactly the same simulation flavour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .analysis.metrics import (
    HD_FEC_RS255_239,
    STRONG_FEC_RS255_213,
    BerResult,
    apply_fec,
    evm_rms,
    measure_ber,
    resolve_rotation,
)
from .dsp.carrier_recovery import BlindPhaseSearch, FrequencyOffsetEstimator
from .dsp.cdc import cd_compensate
from .dsp.equalizer import MimoEqualizer
from .dsp.front_end import CoherentFrontend, matched_filter_and_retime, matched_filter_full_rate
from .physics.amplifier import ErbiumAmplifier
from .physics.channel import SsfmChannel
from .physics.laser import Laser
from .physics.transmitter import CoherentTransmitter
from .utils import Constellation, FibreParams, get_constellation


@dataclass(frozen=True)
class LinkConfig:
    """All user-facing link parameters aggregated in one object.

    Any field can be omitted to keep the library defaults.
    Frozen so every run is deterministic and hashable (Streamlit caching).
    """

    modulation: str = "QPSK"
    symbol_rate: float = 32e9
    sps: int = 4
    roll_off: float = 0.2

    length_km: float = 80.0
    dispersion_ps_per_nm_km: float = 16.0
    alpha_db_km: float = 0.2
    launch_power_dbm: float = 0.0
    enable_nonlinearity: bool = True
    #: number of fibre spans, each followed by an EDFA that restores the
    #: launch power (more spans = more accumulated nonlinear phase for the
    #: same end-of-link OSNR, since the power is re-amplified more often)
    n_spans: int = 1

    tx_linewidth_khz: float = 100.0
    lo_linewidth_khz: float = 100.0
    osnr_db: float = 22.0
    lo_freq_offset_ghz: float = 0.0  # injected crosstalk in the LO (FOE removes it)
    fo_threshold_hz: float = 5e6  # only correct offsets above this (else noise spurs hurt)
    n_symbols: int = 2**14
    seed: int | None = 1234
    #: forward error correction: ``"none"``, ``"hd"`` (7% RS(255,239)) or
    #: ``"strong"`` (~20% RS(255,213)). Post-FEC BER is reported separately.
    fec: str = "none"

    # receiver imperfections
    iq_gain_imbalance_db: float = 0.5
    iq_phase_imbalance_deg: float = 2.0
    adc_bits: int = 8

    # DSP block toggles
    run_cdc: bool = True
    equalizer_taps: int = 15
    mu_cma: float = 1e-3
    mu_mma: float = 1e-4
    n_cma: int | None = 4000
    #: training-preamble length (symbols) for the data-aided LS equalizer init
    n_tr: int = 512
    bps_phases: int = 32
    bps_block: int = 16

    def constellation(self) -> Constellation:
        return get_constellation(self.modulation)


@dataclass
class LinkResult:
    """Bag of intermediate signals and final metrics from a simulation run."""

    config: LinkConfig
    tx_bits_x: NDArray[np.uint8]
    tx_bits_y: NDArray[np.uint8]
    tx_symbols_x: NDArray[np.complex128]
    tx_symbols_y: NDArray[np.complex128]
    tx_field: NDArray[np.complex128]
    rx_field: NDArray[np.complex128]
    rx_wide: NDArray[np.complex128]
    rx_samples: NDArray[np.complex128]
    eq_in: NDArray[np.complex128]
    eq_out: NDArray[np.complex128]
    cr_out: NDArray[np.complex128]
    #: eye-diagram feeds at ``sps`` samples/symbol, stacked (N, 2):
    #: ``eye_pre`` is the raw ADC output (before any DSP block), ``eye_post``
    #: is after CDC + matched filter (open eye, pre-equalizer).
    eye_pre: NDArray[np.complex128]
    eye_post: NDArray[np.complex128]
    fs: float
    equalizer_errors: list[float] = field(default_factory=list)
    evm_percent: tuple[float, float] = (0.0, 0.0)
    ber: BerResult | None = None
    #: post-FEC bit error rate when ``config.fec != "none"`` (else ``None``)
    post_fec_ber: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_link(config: LinkConfig, quiet: bool = False) -> LinkResult:
    """Simulate a full PDM coherent link and return transmitted/received data.

    Pipelined blocks:

    TX (RRC, PDM) -> fiber (SSFM) -> EDFA (target OSNR) -> coherent RX (IQ, ADC)
      -> CDC -> matched filter & retiming -> MIMO equalizer (CMA+MMA)
      -> frequency-offset correction -> BPS -> metrics.
    """
    const = config.constellation()
    fibre = FibreParams(
        wavelength_nm=1550.0,
        alpha_db_km=config.alpha_db_km,
        dispersion_ps=config.dispersion_ps_per_nm_km,
        nonlinear_index_m2_w=(2.6e-20 if config.enable_nonlinearity else 0.0),
        effective_area_um2=80.0,
    )

    tx = CoherentTransmitter(
        constellation=const,
        symbol_rate=config.symbol_rate,
        sps=config.sps,
        beta=config.roll_off,
        power_dbm=config.launch_power_dbm,
        linewidth_khz=config.tx_linewidth_khz,
        seed=config.seed,
    )
    t = tx.transmit(config.n_symbols, seed=config.seed)
    fs = t.fs
    ex = t.ex
    ey = t.ey
    sx, sy = t.sx, t.sy
    bx, by = t.bx, t.by

    channel = SsfmChannel(fibre, dz_km=0.5, manakov=True)

    # Cascaded spans: each span propagates through ``length_km / n_spans`` of
    # fibre, then an EDFA restores the launch power. With a fixed end-of-link
    # OSNR the per-stage ASE budget is split evenly, so the only thing that
    # grows with ``n_spans`` is the accumulated nonlinear phase.
    n_spans = max(1, int(config.n_spans))
    span_km = config.length_km / n_spans
    span_gain_db = config.alpha_db_km * span_km
    base_seed = config.seed if config.seed is not None else 0
    rx_x, rx_y = ex, ey
    for i_span in range(n_spans):
        rx_x, rx_y = channel.propagate(rx_x, rx_y, span_km, fs)
        amp = ErbiumAmplifier(
            gain_db=span_gain_db,
            target_osnr_db=config.osnr_db,
            n_amplifiers=n_spans,
            seed=(base_seed + 1_000_003 * i_span) % (2**31 - 1),
        )
        rx_x, rx_y = amp.amplify(rx_x, rx_y, fs)

    lo = Laser(power_dbm=10.0, linewidth_khz=config.lo_linewidth_khz, seed=config.seed)
    fe = CoherentFrontend(
        lo=lo,
        gain_imbalance_db=config.iq_gain_imbalance_db,
        phase_imbalance_deg=config.iq_phase_imbalance_deg,
        adc_bits=config.adc_bits,
        seed=config.seed,
        lo_freq_offset_hz=config.lo_freq_offset_ghz * 1e9,
    )
    r_x, r_y = fe.detect(rx_x, rx_y, fs)
    eye_pre = np.stack([r_x, r_y], axis=-1)

    if config.run_cdc:
        r_x = cd_compensate(r_x, fs, fibre.beta2, config.length_km, config.alpha_db_km)
        r_y = cd_compensate(r_y, fs, fibre.beta2, config.length_km, config.alpha_db_km)

    # Blind frequency-offset estimation/correction *before* the adaptive
    # stages (standard practice): CMA would otherwise consume some of the
    # carrier rotation and leave residual slips for the phase recovery.
    foe = FrequencyOffsetEstimator()
    fo_est = foe.estimate(r_x, r_y, fs)
    if abs(fo_est) > config.fo_threshold_hz:
        corr_x, corr_y = foe.compensate(r_x, r_y, fo_est, fs)
        assert corr_y is not None
        r_x, r_y = corr_x, corr_y

    eye_post = np.stack(
        [
            matched_filter_full_rate(r_x, config.sps, beta=config.roll_off),
            matched_filter_full_rate(r_y, config.sps, beta=config.roll_off),
        ],
        axis=-1,
    )

    e_x, _ = matched_filter_and_retime(r_x, config.sps, beta=config.roll_off)
    e_y, _ = matched_filter_and_retime(r_y, config.sps, beta=config.roll_off)

    n_sym = min(len(e_x), len(e_y), len(sx))
    e_x, e_y, sx, sy = e_x[:n_sym], e_y[:n_sym], sx[:n_sym], sy[:n_sym]
    bx, by = bx[: n_sym * const.bits_per_symbol], by[: n_sym * const.bits_per_symbol]

    eq = MimoEqualizer(
        n_taps=config.equalizer_taps,
        mu_cma=config.mu_cma,
        mu_mma=config.mu_mma,
        seed=config.seed,
    )
    eq_in = np.stack([e_x, e_y], axis=-1)
    y_x, y_y, eqerr = eq.equalize(
        e_x,
        e_y,
        const,
        n_cma=config.n_cma,
        train_symbols=(sx, sy),
        n_tr=config.n_tr,
    )

    bps = BlindPhaseSearch(const, n_phases=config.bps_phases, block_size=config.bps_block)
    phase_x = bps.estimate(y_x)
    z_x = bps.apply(y_x, phase_x)
    phase_y = bps.estimate(y_y)
    z_y = bps.apply(y_y, phase_y)

    # The T-spaced MIMO equalizer's acausal window introduces a fixed
    # ``c = (n_taps - 1) // 2`` symbol latency; drop it so that the
    # post-BPS symbols are aligned sample-by-sample with the TX references.
    eq_latency = (config.equalizer_taps - 1) // 2
    z_x = z_x[eq_latency:]
    z_y = z_y[eq_latency:]

    # CMA/BPS leave a 2*pi/M phase ambiguity -> resolve it before measuring
    # rotation-free EVM (BER handles it separately inside ``measure_ber``).
    rot_x = resolve_rotation(z_x, const, sx[: len(z_x)])
    rot_y = resolve_rotation(z_y, const, sy[: len(z_y)])
    z_x_rot = z_x * np.exp(1j * 2.0 * np.pi * rot_x / const.symmetry_order)
    z_y_rot = z_y * np.exp(1j * 2.0 * np.pi * rot_y / const.symmetry_order)

    evm0 = evm_rms(z_x_rot[:], sx[: len(z_x_rot)])
    evm1 = evm_rms(z_y_rot[:], sy[: len(z_y_rot)])

    res = LinkResult(
        config=config,
        tx_bits_x=bx,
        tx_bits_y=by,
        tx_symbols_x=sx,
        tx_symbols_y=sy,
        tx_field=ex,
        rx_field=rx_x,
        rx_wide=r_x,
        rx_samples=e_x,
        eye_pre=eye_pre,
        eye_post=eye_post,
        eq_in=eq_in,
        eq_out=np.stack([y_x, y_y], axis=-1),
        cr_out=np.stack([z_x, z_y], axis=-1),
        fs=fs / config.sps,
        equalizer_errors=eqerr,
        evm_percent=(evm0, evm1),
        extra={
            "freq_offset_est_hz": fo_est,
            "n_samples_eq": len(y_x),
            "n_spans": n_spans,
            "bps_phase_deg": np.degrees(phase_x),
            "bps_phase_idx": (np.arange(len(phase_x), dtype=np.float64) + 0.5) * config.bps_block,
        },
    )

    # BER (the 2*pi/M phase ambiguity is per-polarisation, so resolve it on
    # each pol independently, then combine errors over both).
    bit_len = min(len(z_x), len(z_y))
    ber_x = measure_ber(z_x[:bit_len], const, bx[: bit_len * const.bits_per_symbol])
    ber_y = measure_ber(z_y[:bit_len], const, by[: bit_len * const.bits_per_symbol])
    res.ber = BerResult(
        ber=(ber_x.n_errors + ber_y.n_errors) / (ber_x.n_bits + ber_y.n_bits),
        n_errors=ber_x.n_errors + ber_y.n_errors,
        n_bits=ber_x.n_bits + ber_y.n_bits,
        best_rotation=ber_x.best_rotation,
    )

    # Forward error correction: report the post-FEC BER alongside the raw one.
    fec = config.fec.lower()
    if fec not in ("none", "hd", "strong"):
        raise ValueError(f"unknown FEC mode {config.fec!r}")
    if fec != "none" and res.ber is not None:
        code = HD_FEC_RS255_239 if fec == "hd" else STRONG_FEC_RS255_213
        res.post_fec_ber = apply_fec(res.ber.ber, code)
        res.extra["fec"] = code.name
    return res
