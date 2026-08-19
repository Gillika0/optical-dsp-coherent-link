"""Erbium-doped fibre amplifier (EDFA) with ASE noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..utils import C_LIGHT, H_PLANCK, db_to_linear, ref_bandwidth_hz


def bose_einstien_factor(gain_db: float, noise_figure_db: float) -> float:
    """Spontaneous-emission factor ``n_sp`` from gain and noise figure."""
    gain = db_to_linear(gain_db)
    nf = db_to_linear(noise_figure_db)
    return max(float((nf - 1.0) / (2.0 * (1.0 - 1.0 / gain))), 0.0) if gain > 1.0 else 0.0


def ase_noise_power(
    gain_db: float,
    noise_figure_db: float,
    wavelength_m: float,
    bandwidth_hz: float,
) -> float:
    """Total dual-pol ASE power in an optical filter of width ``bandwidth_hz``.

    .. math:: P_\\text{ASE} = 2\\,n_\\text{sp}\\,h\\nu\\,(G-1)\\,B_\\text{opt}
    """
    gain = db_to_linear(gain_db)
    n_sp = bose_einstien_factor(gain_db, noise_figure_db)
    nu = C_LIGHT / wavelength_m
    return 2.0 * n_sp * H_PLANCK * nu * (gain - 1.0) * bandwidth_hz


@dataclass
class ErbiumAmplifier:
    """Noiseless-gain EDFA plus additive Gaussian ASE noise.

    Two operating modes (mutually exclusive):

    * **Gain mode** — ``gain_db`` is fixed and ASE follows from the noise
      figure via the Bose–Einstein factor.
    * **OSNR mode** — the amplifier gain restores the launch power and ASE
      is scaled so the received OSNR exactly equals ``target_osnr_db``
      (referenced to a 0.1 nm resolution bandwidth). With ``n_amplifiers``
      periodic amplifiers (one per span) the ASE is split evenly between
      them, so the end-of-link OSNR still hits the target.

    Attributes
    ----------
    gain_db:
        Small-signal gain.
    noise_figure_db:
        Optical noise figure (gain mode).
    target_osnr_db:
        Desired post-amplifier OSNR (OSNR mode). ``None`` keeps gain mode.
    wavelength_nm:
        ASE carrier wavelength.
    ref_band_nm:
        OSNR reference bandwidth.
    n_amplifiers:
        Number of cascaded amplifiers sharing the total ASE budget (OSNR
        mode). ``1`` keeps the classic single-amplifier behaviour.
    """

    gain_db: float = 20.0
    noise_figure_db: float = 5.0
    target_osnr_db: float | None = None
    wavelength_nm: float = 1550.0
    ref_band_nm: float = 0.1
    n_amplifiers: int = 1
    seed: int | None = 42

    def amplify(
        self,
        ex: NDArray[np.complex128],
        ey: NDArray[np.complex128],
        sample_rate: float,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Amplify both polarisations and add ASE noise."""
        gain = db_to_linear(self.gain_db)
        out_x = np.sqrt(gain) * ex
        out_y = np.sqrt(gain) * ey

        ref_bw = ref_bandwidth_hz(self.wavelength_nm * 1e-9, self.ref_band_nm)

        if self.target_osnr_db is not None:
            # Total dual-pol signal power after the ideal gain.
            p_total = float(np.mean(np.abs(out_x) ** 2 + np.abs(out_y) ** 2))
            osnr_lin = db_to_linear(self.target_osnr_db)
            # With ``n_amplifiers`` cascaded stages the ASE budget is split:
            # each stage contributes n0 so that the *sum* still hits OSNR.
            n_amp = max(1, int(self.n_amplifiers))
            per_pol_n0 = p_total / (2.0 * n_amp * osnr_lin * ref_bw)  # [W/Hz] per stage
            var = max(per_pol_n0 * sample_rate, 0.0)  # variance per sample
        else:
            gain_lin = db_to_linear(self.gain_db)
            n_sp = bose_einstien_factor(self.gain_db, self.noise_figure_db)
            nu = C_LIGHT / (self.wavelength_nm * 1e-9)
            per_pol_n0 = n_sp * H_PLANCK * nu * (gain_lin - 1.0)  # one-sided, per pol
            var = per_pol_n0 * sample_rate

        if var <= 0.0:
            return out_x, out_y

        rng = np.random.default_rng(self.seed)
        sigma = float(np.sqrt(var / 2.0))
        n_x = rng.normal(0.0, sigma, len(out_x)) + 1j * rng.normal(0.0, sigma, len(out_x))
        n_y = rng.normal(0.0, sigma, len(out_y)) + 1j * rng.normal(0.0, sigma, len(out_y))
        return out_x + n_x, out_y + n_y


def add_ase_to_osnr(
    ex: NDArray[np.complex128],
    ey: NDArray[np.complex128],
    target_osnr_db: float,
    sample_rate: float,
    wavelength_nm: float = 1550.0,
    ref_band_nm: float = 0.1,
    n_amplifiers: int = 1,
    seed: int = 42,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Convenience: add white circular Gaussian noise to hit ``target_osnr_db``.

    The signal power is measured from the input fields; no gain is applied.
    ``n_amplifiers`` splits the ASE budget as in ``ErbiumAmplifier``.
    """
    ref_bw = ref_bandwidth_hz(wavelength_nm * 1e-9, ref_band_nm)
    p_total = float(np.mean(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    osnr_lin = db_to_linear(target_osnr_db)
    n_amp = max(1, int(n_amplifiers))
    per_pol_n0 = max(p_total / (2.0 * n_amp * osnr_lin * ref_bw), 0.0)
    var = per_pol_n0 * sample_rate
    if var <= 0.0:
        return ex.copy(), ey.copy()

    rng = np.random.default_rng(seed)
    sigma = float(np.sqrt(var / 2.0))
    n_x = rng.normal(0.0, sigma, len(ex)) + 1j * rng.normal(0.0, sigma, len(ex))
    n_y = rng.normal(0.0, sigma, len(ey)) + 1j * rng.normal(0.0, sigma, len(ey))
    return ex + n_x, ey + n_y
