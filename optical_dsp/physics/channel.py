"""Optical fibre channel: symmetric split-step Fourier method (NLSE)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..utils import FibreParams


@dataclass
class SsfmChannel:
    """Propagate dual-polarisation fields through the NLSE using symmetric SSFM.

    Solves the (Manakov) coupled nonlinear Schrödinger equation

    .. math::
        \\frac{\\partial E_{x,y}}{\\partial z} =
            -\\frac{\\alpha}{2}E_{x,y}
            - j\\frac{\\beta_2}{2}\\frac{\\partial^2 E_{x,y}}{\\partial t^2}
            + j\\gamma\\big(|E_{x,y}|^2+|E_{y,x}|^2\\big)E_{x,y}

    with :math:`\\gamma` optionally scaled by the Manakov :math:`8/9` factor.
    Attenuation and dispersion are applied in the frequency domain via the
    linear operator

    .. math:: \\hat D(\\omega) = -\\alpha/2 + j\\,\\beta_2\\omega^2/2,

    Kerr nonlinearity is applied in the time domain.  One symmetric step is

    .. math::
        E(z+h) \\approx e^{\\hat D h/2}\\, e^{j\\gamma h |E|^2}\\, e^{\\hat D h/2}\\,E(z).

    Attributes
    ----------
    fibre:
        Physical fibre parameters (:math:`\\alpha, \\beta_2, \\gamma`).
    dz_km:
        Spatial step size.
    manakov:
        Use the 8/9 Manakov coefficient for XPM.
    """

    fibre: FibreParams = FibreParams()
    dz_km: float = 0.5
    manakov: bool = True

    _cache_key: tuple[int, float, float] | None = field(default=None, repr=False)
    _lin_half: NDArray[np.complex128] | None = field(default=None, repr=False)
    _lin_full: NDArray[np.complex128] | None = field(default=None, repr=False)
    _n_steps: int | None = field(default=None, repr=False)

    @property
    def gamma(self) -> float:
        """Effective Kerr coefficient (Manakov-scaled when enabled)."""
        g = self.fibre.gamma
        return (8.0 / 9.0) * g if self.manakov else g

    def _prepare(self, n_samples: int, sample_rate: float, length_km: float) -> int:
        """Cache frequency grids and linear operators for ``(n, fs, L)``."""
        n_steps = max(2, int(np.round(length_km / self.dz_km)))
        key = (n_samples, float(sample_rate), float(length_km))
        if self._cache_key != key:
            freq = np.fft.fftfreq(n_samples, d=1.0 / sample_rate)
            omega = 2.0 * np.pi * freq
            alpha_lin = (self.fibre.alpha_db_km / 10.0 * np.log(10.0)) / 1e3  # dB/km -> 1/m
            beta2 = self.fibre.beta2
            dz = length_km / n_steps * 1e3  # in metres
            self._cache_key = key
            self._freq = freq
            self._n_steps = n_steps
            self._lin_half = np.exp((-alpha_lin / 2.0 + 1j * beta2 / 2.0 * omega**2) * (dz / 2.0))
            self._lin_full = np.exp((-alpha_lin / 2.0 + 1j * beta2 / 2.0 * omega**2) * dz)
        return n_steps

    def propagate(
        self,
        ex: NDArray[np.complex128],
        ey: NDArray[np.complex128],
        length_km: float,
        sample_rate: float,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Propagate both polarisations over ``length_km`` of fibre."""
        assert len(ex) == len(ey)
        n_samples = len(ex)
        n_steps = self._prepare(n_samples, sample_rate, length_km)
        gamma = self.gamma
        dz = length_km / n_steps * 1e3  # [m] inside the exponents

        ex_f = np.fft.fft(ex)
        ey_f = np.fft.fft(ey)
        lin_half = self._lin_half
        assert lin_half is not None
        for _ in range(n_steps):
            ex_f = lin_half * ex_f
            ey_f = lin_half * ey_f
            ex_t = np.fft.ifft(ex_f)
            ey_t = np.fft.ifft(ey_f)

            intensity = np.abs(ex_t) ** 2 + np.abs(ey_t) ** 2
            nl = np.exp(1j * gamma * intensity * dz)
            ex_t = ex_t * nl
            ey_t = ey_t * nl

            ex_f = np.fft.fft(ex_t)
            ey_f = np.fft.fft(ey_t)
            ex_f = lin_half * ex_f
            ey_f = lin_half * ey_f

        return np.fft.ifft(ex_f), np.fft.ifft(ey_f)


# --------------------------------------------------------------------------- #
#  Analytical references (used for verification and theory curves)
# --------------------------------------------------------------------------- #


def analytical_dispersion_transfer(
    n_samples: int, sample_rate: float, length_km: float, fibre: FibreParams
) -> NDArray[np.complex128]:
    """Frequency-domain transfer function for the linear channel only.

    .. math:: H(\\omega, L) = \\exp\\bigl(-\\alpha L/2 + j\\beta_2\\omega^2 L/2\\bigr)
    """
    freq = np.fft.fftfreq(n_samples, d=1.0 / sample_rate)
    omega = 2.0 * np.pi * freq
    alpha_lin = (fibre.alpha_db_km / 10.0 * np.log(10.0)) / 1e3
    length_m = length_km * 1e3
    h: NDArray[np.complex128] = np.exp(
        (-alpha_lin / 2.0 + 1j * fibre.beta2 / 2.0 * omega**2) * length_m
    )
    return h


def apply_analytical_linear_channel(
    sig: NDArray[np.complex128],
    sample_rate: float,
    length_km: float,
    fibre: FibreParams,
) -> NDArray[np.complex128]:
    """Propagate a single-pol signal through the exact linear channel."""
    h = analytical_dispersion_transfer(len(sig), sample_rate, length_km, fibre)
    return np.fft.ifft(np.fft.fft(sig) * h)
