"""Coherent transmitter: PRBS, Gray-coded M-QAM mapping, RRC pulse shaping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..utils import Constellation, dbm_to_w

# --------------------------------------------------------------------------- #
#  PRBS (linear-feedback shift register)
# --------------------------------------------------------------------------- #


def generate_prbs(
    n_bits: int, poly: tuple[int, ...] = (7, 6), seed: int | None = None
) -> NDArray[np.uint8]:
    """Generate ``n_bits`` pseudo-random bits with an LFSR.

    ``poly`` describes the generator polynomial taps (after the LFSR's top bit),
    e.g. PRBS7 ``= x^7 + x^6 + 1`` -> ``poly=(7,6)``.
    """
    order = poly[0]
    rng = np.random.default_rng(seed)
    state = int(rng.integers(1, 2**order))
    out = np.empty(n_bits, dtype=np.uint8)
    feedback_taps = poly[1:]
    for i in range(n_bits):
        out[i] = np.uint8(state & 1)
        bit = state & 1
        state >>= 1
        if bit:
            state ^= sum(1 << (order - t) for t in feedback_taps)
    return out


def gray_map_bits(bits: NDArray[np.uint8], constellation: Constellation) -> NDArray[np.complex128]:
    """Map a bit stream to unit-energy constellation symbols (vectorised)."""
    return constellation.bits_to_symbols(bits)


# --------------------------------------------------------------------------- #
#  RRC pulse shaping
# --------------------------------------------------------------------------- #


def rrc_taps(beta: float, sps: int, n_taps: int, normalize: bool = True) -> NDArray[np.float64]:
    """Root-raised-cosine impulse response (samples).

    Closed form (Proakis) evaluated at integer sample offsets:

    .. math::
        h(t) &=
            \\frac{\\sin\\pi(1-\\beta)t/T
                   + 4\\beta(t/T)\\cos\\pi(1+\\beta)t/T}
                 {\\pi(t/T)\\big(1-(4\\beta t/T)^2\\big)}

    with the standard special cases at :math:`t=0` and :math:`t=\\pm T/4\\beta`.

    Parameters
    ----------
    beta:
        Roll-off factor in [0, 1].
    sps:
        Samples per symbol.
    n_taps:
        Filter order in symbols (must be odd).
    normalize:
        Normalise the taps so their squared sum equals ``sps`` (unit energy
        per symbol: a unit-power symbol stream keeps unit power after
        pulse shaping, so launch-power budgets are exact).
    """
    assert 0.0 <= beta <= 1.0
    assert n_taps % 2 == 1, "odd number of taps required"
    taps = np.arange(n_taps, dtype=np.float64) - (n_taps - 1) / 2.0
    x = taps / sps  # in symbol periods
    h = np.zeros(n_taps)

    i_zero = np.isclose(x, 0.0)
    i_special = np.zeros(n_taps, dtype=bool)
    if beta > 0.0:
        i_special = np.isclose(np.abs(x), 1.0 / (4.0 * beta))
    i_general = ~(i_zero | i_special)

    h[i_zero] = 1.0 - beta + 4.0 * beta / np.pi
    if beta > 0.0:
        c = 1.0 / np.sqrt(2.0)
        h[i_special] = (
            beta
            * c
            * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        )

    xg = x[i_general]
    num = np.sin(np.pi * (1.0 - beta) * xg) + 4.0 * beta * xg * np.cos(np.pi * (1.0 + beta) * xg)
    den = np.pi * xg * (1.0 - (4.0 * beta * xg) ** 2)
    h[i_general] = num / den

    if normalize:
        energy = float(np.sum(h**2))
        assert energy > 0.0
        h = h * np.sqrt(float(sps) / energy)
    return h


# --------------------------------------------------------------------------- #
#  Transmitter
# --------------------------------------------------------------------------- #


@dataclass
class TransmitFrame:
    """Typed handle to the transmitter output (fields + references)."""

    ex: NDArray[np.complex128]
    ey: NDArray[np.complex128]
    sx: NDArray[np.complex128]
    sy: NDArray[np.complex128]
    bx: NDArray[np.uint8]
    by: NDArray[np.uint8]
    fs: float
    sps: int


@dataclass
class CoherentTransmitter:
    """PDM coherent transmitter.

    Emits polarisation-multiplexed, root-raised-cosine shaped, unit-power
    modulated fields carrying a common carrier (shared TX laser phase).

    Attributes
    ----------
    constellation:
        Modulation format.
    symbol_rate:
        Baud rate in [Bd].
    sps:
        Samples per symbol (integer oversampling factor).
    beta:
        RRC roll-off factor.
    rrc_taps_sym:
        RRC filter length in symbols.
    power_dbm:
        Total dual-pol launch power.
    linewidth_khz:
        TX laser linewidth.
    wavelength_nm:
        Carrier wavelength.
    seed:
        Global RNG seed.
    """

    constellation: Constellation
    symbol_rate: float
    sps: int = 4
    beta: float = 0.2
    rrc_taps_sym: int = 33
    power_dbm: float = 0.0
    linewidth_khz: float = 100.0
    wavelength_nm: float = 1550.0
    seed: int | None = 1

    _shape_filter: NDArray[np.float64] | None = None

    @property
    def sample_rate(self) -> float:
        """DAC/ADC sampling frequency in samples/s."""
        return self.symbol_rate * self.sps

    @property
    def samples_per_symbol(self) -> int:
        return self.sps

    def shape_filter(self) -> NDArray[np.float64]:
        """(Cached) transmit RRC impulse response."""
        if self._shape_filter is None:
            self._shape_filter = rrc_taps(self.beta, self.sps, self.rrc_taps_sym)
        return self._shape_filter

    def transmit(self, n_symbols: int = 2**14, seed: int | None = None) -> TransmitFrame:
        """Generate a dual-pol (X/Y) optical field plus references."""
        rng = np.random.default_rng(self.seed if seed is None else seed)

        bps = self.constellation.bits_per_symbol
        n_bits = n_symbols * bps
        # Uniform random payload (uniform constellations: a LFSR PRBS period
        # like 127 is not divisible by 2/4/6-bit groups and would bias the
        # symbol histogram, skewing per-pol powers and EVM).
        bits_x = rng.integers(0, 2, n_bits, dtype=np.uint8)
        bits_y = rng.integers(0, 2, n_bits, dtype=np.uint8)

        sym_x = gray_map_bits(bits_x, self.constellation)
        sym_y = gray_map_bits(bits_y, self.constellation)

        field_x = self._shape_and_scale(sym_x)
        field_y = self._shape_and_scale(sym_y)

        phi = phase_noise_common(
            n_samples=len(field_x),
            fs=self.sample_rate,
            linewidth_hz=self.linewidth_khz * 1e3,
            seed=int(rng.integers(0, 1 << 30)),
        )
        carrier = np.exp(1j * phi)
        field_x = field_x * carrier
        field_y = field_y * carrier

        return TransmitFrame(
            ex=field_x,
            ey=field_y,
            sx=sym_x,
            sy=sym_y,
            bx=bits_x,
            by=bits_y,
            fs=self.sample_rate,
            sps=self.sps,
        )

    def _shape_and_scale(self, symbols: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Upsample, shape and apply per-polarisation launch power."""
        h = self.shape_filter()
        sps = self.sps
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
        upsampled[::sps] = symbols
        conv = np.convolve(upsampled, h)
        center = (len(h) - 1) // 2
        shaped = conv[center : center + len(upsampled)]
        per_pol_power = 0.5 * dbm_to_w(self.power_dbm)
        out: NDArray[np.complex128] = np.sqrt(per_pol_power) * shaped
        return out


def phase_noise_common(
    n_samples: int, fs: float, linewidth_hz: float, seed: int | None
) -> NDArray[np.float64]:
    """Shared Wiener phase for both polarisations (single TX laser)."""
    from .laser import phase_noise_walk

    phase: NDArray[np.float64] = phase_noise_walk(n_samples, fs, linewidth_hz, seed)
    return phase
