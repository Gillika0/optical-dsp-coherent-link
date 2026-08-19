"""CW laser source with Lorentzian linewidth (Wiener phase noise)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..utils import C_LIGHT, dbm_to_w


def phase_noise_walk(
    n_samples: int, sample_rate: float, linewidth_hz: float, seed: int | None = None
) -> NDArray[np.float64]:
    """Wiener-phase-noise realisation (Lorentzian laser linewidth).

    The instantaneous phase obeys

    .. math:: \\phi_{k+1} = \\phi_k + \\sqrt{2\\pi\\,\\Delta\\nu\\,\\Delta t}\\,n_k,
               \\quad n_k \\sim \\mathcal N(0,1).

    Returns the unwrapped phase trajectory in radians.
    """
    rng = np.random.default_rng(seed)
    step_sigma = float(np.sqrt(2.0 * np.pi * linewidth_hz / sample_rate))
    increments = rng.normal(0.0, step_sigma, n_samples)
    return np.cumsum(increments)


@dataclass(frozen=True)
class Laser:
    """Ideal CW laser source.

    Attributes
    ----------
    power_dbm:
        Optical launch power in [dBm].
    linewidth_khz:
        Full-width at half-maximum Lorentzian linewidth :math:`\\Delta\\nu`.
    wavelength_nm:
        Vacuum wavelength of the carrier.
    seed:
        RNG seed (None = non-deterministic).
    """

    power_dbm: float = 0.0
    linewidth_khz: float = 100.0
    wavelength_nm: float = 1550.0
    seed: int | None = None

    @property
    def power_w(self) -> float:
        """Output power in Watts."""
        return dbm_to_w(self.power_dbm)

    @property
    def frequency_hz(self) -> float:
        """Carrier optical frequency in Hz."""
        return C_LIGHT / (self.wavelength_nm * 1e-9)

    def field(
        self, n_samples: int, sample_rate: float, seed: int | None = None
    ) -> NDArray[np.complex128]:
        """Complex envelope of the CW field (power-normalised).

        ``mean(|E|^2) == power_w``.
        """
        rng_seed = self.seed if seed is None else seed
        phi = phase_noise_walk(n_samples, sample_rate, self.linewidth_khz * 1e3, rng_seed)
        out: NDArray[np.complex128] = np.sqrt(self.power_w) * np.exp(1j * phi)
        return out
