"""Carrier recovery: 4th-power FOE and blind phase search (BPS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..utils import Constellation

# --------------------------------------------------------------------------- #
#  Frequency offset estimation (4th-power FFT)
# --------------------------------------------------------------------------- #


def estimate_frequency_offset(sig: NDArray[np.complex128], sample_rate: float) -> float:
    """Coarse frequency-offset estimate via the 4th-power method.

    .. math:: \\hat f_\\Delta = \\frac{1}{4}\\,\\arg\\max_{\\omega}
              \\big|\\mathcal F\\{r^4\\}(\\omega)\\big|
    """
    n = len(sig)
    if n == 0:
        return 0.0
    spectrum = np.fft.fftshift(np.fft.fft(sig**4))
    peak_index = int(np.argmax(np.abs(spectrum)))
    f_res = sample_rate / n
    f_peak = (peak_index - n // 2) * f_res
    return f_peak / 4.0


def remove_frequency_offset(
    sig: NDArray[np.complex128], freq_offset_hz: float, sample_rate: float
) -> NDArray[np.complex128]:
    """De-rotate the signal by ``freq_offset_hz`` (vectorised)."""
    n = len(sig)
    t = np.arange(n, dtype=np.float64) / sample_rate
    out: NDArray[np.complex128] = sig * np.exp(-1j * 2.0 * np.pi * freq_offset_hz * t)
    return out


@dataclass
class FrequencyOffsetEstimator:
    """Estimate and compensate the common carrier frequency offset (PDM)."""

    def estimate(
        self,
        sig0: NDArray[np.complex128],
        sig1: NDArray[np.complex128] | None = None,
        sample_rate: float = 1.0,
    ) -> float:
        """Average the per-polarisation 4th-power estimates."""
        fo = estimate_frequency_offset(sig0, sample_rate)
        if sig1 is not None and len(sig1) > 0:
            fo = 0.5 * (fo + estimate_frequency_offset(sig1, sample_rate))
        return float(fo)

    def compensate(
        self,
        sig0: NDArray[np.complex128],
        sig1: NDArray[np.complex128] | None,
        freq_offset_hz: float,
        sample_rate: float,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]:
        """De-rotate both polarisations by ``freq_offset_hz``."""
        out0 = remove_frequency_offset(sig0, freq_offset_hz, sample_rate)
        out1 = (
            remove_frequency_offset(sig1, freq_offset_hz, sample_rate) if sig1 is not None else None
        )
        return out0, out1


# --------------------------------------------------------------------------- #
#  Blind phase search (BPS)
# --------------------------------------------------------------------------- #


@dataclass
class BlindPhaseSearch:
    """Block-based BPS carrier-phase estimation.

    For every block of symbols the squared Euclidean distance to the
    constellation is summed over a grid of ``n_phases`` test rotations
    within one quadrant; the ArgMin phase is selected and the block-wise
    phases are unwrapped (in the M-fold domain) to avoid quadrant jumps.

    Attributes
    ----------
    constellation:
        Reference modulation format (unit-energy symbols).
    n_phases:
        Number of test phases per quadrant.
    block_size:
        Symbols per phase-decision block.
    """

    constellation: Constellation
    n_phases: int = 32
    block_size: int = 64

    def __post_init__(self) -> None:
        assert self.n_phases >= 1 and self.block_size >= 1
        self._sym = self.constellation.symbols.astype(np.complex128)
        self._m = self.constellation.symmetry_order

    def _test_phases(self) -> NDArray[np.float64]:
        step = 2.0 * np.pi / self._m / self.n_phases
        return (np.arange(self.n_phases, dtype=np.float64) + 0.5) * step

    def run(self, sym: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply BPS to a symbol-spaced stream; returns the phase-corrected data.

        The output retains an arbitrary global rotation multiple of
        ``2*pi/M`` (resolved downstream, e.g. in BER counting).
        """
        phases = self._test_phases()
        n = len(sym)
        if n == 0:
            return sym.copy()

        m = self._m
        block_phase = np.zeros(((n + self.block_size - 1) // self.block_size,), dtype=np.float64)
        for b, start in enumerate(range(0, n, self.block_size)):
            block = sym[start : start + self.block_size]
            rotated = block[:, None] * np.exp(-1j * phases)[None, :]  # (Nb, B)
            dist = np.abs(rotated[:, :, None] - self._sym[None, None, :]) ** 2  # (Nb,B,M)
            cost = dist.min(axis=2).mean(axis=0)  # (B,)
            block_phase[b] = phases[int(np.argmin(cost))]

        # Unwrap in the M-fold domain (2*pi/M period) to stitch quadrants.
        phase_m = block_phase * m
        phase_m = np.unwrap(phase_m, period=2.0 * np.pi)
        phase_unwrapped = phase_m / m

        out = sym.copy()
        for b, start in enumerate(range(0, n, self.block_size)):
            stop = min(start + self.block_size, n)
            out[start:stop] *= np.exp(-1j * phase_unwrapped[b])
        return out
