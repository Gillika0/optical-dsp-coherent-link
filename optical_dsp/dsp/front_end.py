"""Coherent optical front end: 90° hybrid, balanced PDs, ADC quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..physics.laser import Laser
from ..physics.transmitter import rrc_taps


def apply_iq_imbalance(
    sig: NDArray[np.complex128],
    gain_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0,
) -> NDArray[np.complex128]:
    """Inject receiver IQ imbalance.

    Model (self-consistent, identity for zero imbalance):

    .. math::
        I' &= (1+\\varepsilon/2)\\,\\Re\\{s\\} \\\\
        Q' &= (1-\\varepsilon/2)\\big(-\\sin\\varphi\\,\\Re\\{s\\}
              + \\cos\\varphi\\,\\Im\\{s\\}\\big)

    with :math:`\\varepsilon` derived from ``gain_imbalance_db`` and
    :math:`\\varphi` from ``phase_imbalance_deg``.
    """
    if gain_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
        return sig.copy()
    eps = 10.0 ** (gain_imbalance_db / 20.0) - 1.0
    phi = np.deg2rad(phase_imbalance_deg)
    i = (1.0 + eps / 2.0) * sig.real
    q = (1.0 - eps / 2.0) * (-np.sin(phi) * sig.real + np.cos(phi) * sig.imag)
    out: NDArray[np.complex128] = i + 1j * q
    return out


def adc_quantize(
    sig: NDArray[np.complex128], bits: int = 8, clip_sigma: float = 3.5
) -> NDArray[np.complex128]:
    """Uniform ADC quantization (I/Q independently) with soft clipping.

    ``clip_sigma`` sets the full-scale as a multiple of the RMS amplitude.
    """
    assert bits >= 1
    levels = 2 ** (bits - 1)
    rms_i = float(np.sqrt(np.mean(np.abs(sig.real) ** 2)))
    rms_q = float(np.sqrt(np.mean(np.abs(sig.imag) ** 2)))
    scale_i = clip_sigma * rms_i if rms_i > 0 else 1.0
    scale_q = clip_sigma * rms_q if rms_q > 0 else 1.0
    q_i = np.clip(np.round(sig.real / scale_i * levels), -levels, levels - 1) * (scale_i / levels)
    q_q = np.clip(np.round(sig.imag / scale_q * levels), -levels, levels - 1) * (scale_q / levels)
    out: NDArray[np.complex128] = q_i + 1j * q_q
    return out


@dataclass
class CoherentFrontend:
    """Ideal 90°-hybrid coherent receiver with LO and digitization.

    Converts the received optical field into a baseband complex photocurrent:

    .. math:: r(t) = \\rho\\,R\\,E_s(t)\\,E_\\text{LO}^*(t)

    optionally distorts it with receiver IQ imbalance and quantizes it with an
    ``enob``-bit ADC.

    Attributes
    ----------
    lo:
        Local-oscillator laser (linewidth, power).
    responsivity:
        Photodiode responsivity [A/W].
    hybrid_loss:
        Loss/gain factor of the 90° hybrid (+ balanced pair).
    gain_imbalance_db / phase_imbalance_deg:
        Receiver IQ imbalance.
    adc_bits:
        Effective ADC resolution. ``None`` = ideal (no quantization).
    lo_freq_offset_hz:
        Frequency detuning of the LO vs the transmitter carrier. It shows up
        as a rotating carrier on the photocurrent and is cancelled by the
        digital frequency-offset estimator.
    """

    lo: Laser = Laser(power_dbm=10.0, linewidth_khz=100.0)
    responsivity: float = 1.0
    hybrid_loss: float = 1.0
    gain_imbalance_db: float = 0.0
    phase_imbalance_deg: float = 0.0
    adc_bits: int | None = 8
    lo_freq_offset_hz: float = 0.0
    seed: int | None = 7

    def detect(
        self,
        ex: NDArray[np.complex128],
        ey: NDArray[np.complex128],
        sample_rate: float,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Optical-to-electrical conversion for both polarisations."""
        n = len(ex)
        lo_field = self.lo.field(n, sample_rate, seed=self.seed)
        if self.lo_freq_offset_hz != 0.0:
            t = np.arange(n, dtype=np.float64) / sample_rate
            lo_field = lo_field * np.exp(-1j * 2.0 * np.pi * self.lo_freq_offset_hz * t)
        scale = self.responsivity * self.hybrid_loss
        r_x = scale * ex * np.conj(lo_field)
        r_y = scale * ey * np.conj(lo_field)

        has_imbalance = self.gain_imbalance_db != 0.0 or self.phase_imbalance_deg != 0.0
        if has_imbalance:
            r_x = apply_iq_imbalance(r_x, self.gain_imbalance_db, self.phase_imbalance_deg)
            r_y = apply_iq_imbalance(r_y, self.gain_imbalance_db, self.phase_imbalance_deg)
        if self.adc_bits:
            r_x = adc_quantize(r_x, self.adc_bits)
            r_y = adc_quantize(r_y, self.adc_bits)
        return r_x, r_y


def matched_filter_and_retime(
    sig: NDArray[np.complex128],
    sps: int,
    beta: float = 0.2,
    n_taps_sym: int = 33,
    selection: str = "peak",
) -> tuple[NDArray[np.complex128], int]:
    """Matched RRC filtering plus symbol-spaced retiming at 1 sample/symbol.

    ``selection`` chooses the sample offset within the ``sps`` grid:
    ``"peak"`` = index with maximum short-term power; ``0..sps-1`` fixed.
    """
    h = rrc_taps(beta, sps, n_taps_sym)
    conv = np.convolve(sig, h, mode="full")
    center = (len(h) - 1) // 2
    conv = conv[center : center + len(sig)]

    n_sym = len(sig) // sps
    if selection == "peak":
        frames = conv[: n_sym * sps].reshape(n_sym, sps)
        best = int(np.argmax(np.mean(np.abs(frames) ** 2, axis=0)))
    else:
        best = int(selection) % sps
    return conv[best : n_sym * sps : sps], best
