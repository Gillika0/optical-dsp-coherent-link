"""Shared constants, unit conversions and constellation utilities.

All conversions here are vectorised NumPy operations and strictly typed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

C_LIGHT: float = 299_792_458.0  # [m/s]
H_PLANCK: float = 6.626_070_15e-34  # [J*s]

# --------------------------------------------------------------------------- #
#  Unit conversions
# --------------------------------------------------------------------------- #


def dbm_to_w(p: float) -> float:
    """Convert optical power in dBm to Watts."""
    w: float = 10.0 ** (p / 10.0) * 1e-3
    return w


def w_to_dbm(p: float) -> float:
    """Convert optical power in Watts to dBm."""
    db: float = 10.0 * float(np.log10(max(p, 1e-30))) + 30.0
    return db


def db_to_linear(x: float) -> float:
    """Convert dB to a linear ratio."""
    lin: float = 10.0 ** (x / 10.0)
    return lin


def linear_to_db(x: float) -> float:
    """Convert a linear ratio to dB."""
    db: float = 10.0 * float(np.log10(max(x, 1e-30)))
    return db


def beta2_from_D(wavelength_m: float, D_ps: float) -> float:
    """Group-velocity dispersion :math:`\\beta_2`.

    ``D`` is given in [ps/(nm*km)], :math:`\\lambda` in [m].
    Returns :math:`\\beta_2` in [s^2/m].

    .. math:: \\beta_2 = -\\frac{\\lambda^2}{2\\pi c} D
    """
    return -(wavelength_m**2) / (2.0 * np.pi * C_LIGHT) * D_ps * 1e-6


def gamma_from_n2(wavelength_m: float, n2: float, aeff_m2: float) -> float:
    """Kerr nonlinear coefficient :math:`\\gamma` in [1/(W*m)]."""
    return 2.0 * np.pi * n2 / (wavelength_m * aeff_m2)


def ref_bandwidth_hz(wavelength_m: float, band_nm: float = 0.1) -> float:
    """Resolution bandwidth of an optical filter in Hz (e.g. 0.1 nm)."""
    return C_LIGHT * band_nm * 1e-9 / (wavelength_m**2)


def osnr_db_to_snr_db(osnr_db: float, symbol_rate: float, ref_bw_hz: float, npol: int = 1) -> float:
    """Map OSNR to per-polarisation symbol SNR ``Es/N0`` in dB.

    Consistent with ``ErbiumAmplifier``'s convention, which sets the per-pol
    ASE density to ``n0 = Ps / (OSNR * B_ref)``: the per-pol symbol SNR at the
    symbol rate is then ``Es/N0 = OSNR * B_ref / Rs`` (no extra factor).
    ``npol`` only matters if the caller uses a *dual-polarisation-noise* OSNR
    definition (SNR then loses ``10*log10(npol)``).
    """
    snr_lin = db_to_linear(osnr_db) * ref_bw_hz / symbol_rate / npol
    snr_db: float = 10.0 * float(np.log10(max(snr_lin, 1e-30)))
    return snr_db


# --------------------------------------------------------------------------- #
#  Modulation formats (Gray-coded M-QAM constellations)
# --------------------------------------------------------------------------- #


def _gray_seq(n: int) -> NDArray[np.int64]:
    """Binary-reflected Gray sequence of length ``n``."""
    idx = np.arange(n, dtype=np.int64)
    return idx ^ (idx >> 1)


@dataclass(frozen=True)
class Constellation:
    """A unit-energy Gray-coded M-QAM constellation (square or rectangular).

    Symbols are normalised so that ``E[|s|^2] == 1``. Bit labels use the
    binary value ``n`` for a position whose index within each dimension is
    the n-th Gray word, MSB-first across the symbol.

    Parameters
    ----------
    order:
        Number of constellation points (4, 8, 16, 64, ...).
    name:
        Optional human-readable label.
    dims:
        Per-dimension level counts ``(n_i, n_q)``. ``None`` selects the square
        ``(sqrt(order), sqrt(order))`` lattice; pass e.g. ``(2, 4)`` for the
        rectangular 8-QAM cross.
    """

    order: int
    name: str = "M-QAM"
    dims: tuple[int, int] | None = None

    sqrt_order: int = field(init=False)
    bits_per_symbol: int = field(init=False)
    _levels_i: NDArray[np.float64] = field(init=False)
    _levels_q: NDArray[np.float64] = field(init=False)
    _gray_i: NDArray[np.int64] = field(init=False)
    _gray_q: NDArray[np.int64] = field(init=False)
    _pos_i: NDArray[np.int64] = field(init=False)
    _pos_q: NDArray[np.int64] = field(init=False)
    _scale: float = field(init=False)

    #: 1-D array of the ``order`` complex symbols, unit average power.
    symbols: NDArray[np.complex128] = field(init=False, compare=False)
    #: Integer array (``order``, ``bits_per_symbol``) of bit labels.
    bitmap: NDArray[np.uint8] = field(init=False, compare=False)

    _cache: ClassVar[dict[int, Constellation]] = {}

    def __post_init__(self) -> None:
        if self.dims is None:
            root = int(round(np.sqrt(self.order)))
            assert root * root == self.order, "only square QAM constellations"
            dims = (root, root)
        else:
            dims = (int(self.dims[0]), int(self.dims[1]))
            assert dims[0] * dims[1] == self.order, "dims product must equal order"
        sp_i = int(np.log2(dims[0]))
        sp_q = int(np.log2(dims[1]))
        bps = sp_i + sp_q

        levels = [2.0 * np.arange(n_d, dtype=np.float64) - (n_d - 1.0) for n_d in dims]
        grays = [_gray_seq(n_d) for n_d in dims]
        poss = [np.empty(n_d, np.int64) for n_d in dims]
        for d, g in enumerate(grays):
            poss[d][g] = np.arange(dims[d], dtype=np.int64)

        symbols = np.zeros(self.order, dtype=np.complex128)
        bitmap = np.zeros((self.order, bps), dtype=np.uint8)
        for n in range(self.order):
            i_label, q_label = n // dims[1], n % dims[1]
            p_i, p_q = int(poss[0][i_label]), int(poss[1][q_label])
            symbols[n] = levels[0][p_i] + 1j * levels[1][p_q]
            bits_bin = np.unpackbits(np.array([n], dtype=np.uint8))[-bps:]
            bitmap[n] = bits_bin

        scale = float(np.sqrt(np.mean(np.abs(symbols) ** 2)))
        symbols = symbols / scale

        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "sqrt_order", dims[0])
        object.__setattr__(self, "bits_per_symbol", bps)
        object.__setattr__(self, "_levels_i", levels[0] / scale)
        object.__setattr__(self, "_levels_q", levels[1] / scale)
        object.__setattr__(self, "_gray_i", grays[0])
        object.__setattr__(self, "_gray_q", grays[1])
        object.__setattr__(self, "_pos_i", poss[0])
        object.__setattr__(self, "_pos_q", poss[1])
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "bitmap", bitmap)

    # -- bit interleaving helpers -------------------------------------------- *
    def bits_to_symbols(self, bits: NDArray[np.uint8]) -> NDArray[np.complex128]:
        """Map a bit stream (multiple of ``bits_per_symbol`` long) to symbols."""
        bps = self.bits_per_symbol
        n_sym = len(bits) // bps
        chunks = bits[: n_sym * bps].reshape(n_sym, bps)
        idx = np.zeros(n_sym, dtype=np.int64)
        for k in range(bps):
            idx = (idx << 1) | chunks[:, k].astype(np.int64)
        out: NDArray[np.complex128] = self.symbols[idx]
        return out

    def symbols_to_bits(self, idx: NDArray[np.int64]) -> NDArray[np.uint8]:
        """Bit labels for constellation-point indices (vectorised, per symbol)."""
        return self.bitmap[idx].reshape(-1)

    # -- position<->level maps ----------------------------------------------- *
    @staticmethod
    def _demap_dim(
        samples: NDArray[np.float64], levels: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Nearest level decision per dimension (+ squared distance)."""
        dist = (samples[:, None] - levels[None, :]) ** 2
        pos = np.argmin(dist, axis=1)
        return pos, dist[np.arange(len(samples)), pos]

    @staticmethod
    def _position_to_label(pos: NDArray[np.int64], gray: NDArray[np.int64]) -> NDArray[np.int64]:
        """Amplitude-sorted position within a dimension -> Gray bit label."""
        return gray[pos]

    def nearest_index(self, samples: NDArray[np.complex128]) -> NDArray[np.int64]:
        """Nearest constellation-point index for each received sample."""
        i_pos, _ = self._demap_dim(samples.real, self._levels_i)
        q_pos, _ = self._demap_dim(samples.imag, self._levels_q)
        n_q = self.dims[1]
        return self._position_to_label(i_pos, self._gray_i) * n_q + self._position_to_label(
            q_pos, self._gray_q
        )

    def symbol_error_distance(
        self, samples: NDArray[np.complex128]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Nearest (label, squared distance) before constellation scaling."""
        i_pos, d_i = self._demap_dim(samples.real, self._levels_i)
        q_pos, d_q = self._demap_dim(samples.imag, self._levels_q)
        n_q = self.dims[1]
        idx = self._position_to_label(i_pos, self._gray_i) * n_q + self._position_to_label(
            q_pos, self._gray_q
        )
        return idx, d_i + d_q

    @property
    def symmetry_order(self) -> int:
        """Rotational symmetry of the QAM lattice (rotations in 2*pi/k).

        Square M-QAM is invariant under quarter turns; the rectangular
        cross-QAM (2 x 4) has no rotational symmetry, so only the identity
        rotation is explored.
        """
        if self.dims[0] == self.dims[1]:
            return 4
        return 1

    @property
    def decision_radius2_i(self) -> float:
        """MMA second-order modulus for the in-phase dimension ``E[Re(s)^4]/E[Re(s)^2]``."""
        return float(np.mean(self.symbols.real**4) / np.mean(self.symbols.real**2))

    @property
    def decision_radius2_q(self) -> float:
        """MMA second-order modulus for the quadrature dimension."""
        return float(np.mean(self.symbols.imag**4) / np.mean(self.symbols.imag**2))

    @property
    def cma_radius2(self) -> float:
        """CMA dispersion constant ``E[|s|^4]/E[|s|^2]``."""
        return float(np.mean(np.abs(self.symbols) ** 4) / np.mean(np.abs(self.symbols) ** 2))


def _make(order: int, name: str) -> Constellation:
    if order not in Constellation._cache:
        Constellation._cache[order] = Constellation(order=order, name=name)
    return Constellation._cache[order]


def QPSK() -> Constellation:
    """QPSK (order 4) constellation, cached."""
    return _make(4, "QPSK")


def QAM8() -> Constellation:
    """8-QAM constellation (rectangular 2 x 4 cross), cached."""
    if 8 not in Constellation._cache:
        Constellation._cache[8] = Constellation(order=8, name="8-QAM", dims=(2, 4))
    return Constellation._cache[8]


def QAM16() -> Constellation:
    """16-QAM constellation, cached."""
    return _make(16, "16-QAM")


def QAM64() -> Constellation:
    """64-QAM constellation, cached."""
    return _make(64, "64-QAM")


def get_constellation(name: str) -> Constellation:
    """Resolve a modulation-format name to a cached :class:`Constellation`.

    Accepts plain formats (``QPSK``, ``8QAM``, ``16-QAM``, ``64QAM``) and
    their dual-polarisation aliases (``DP-QPSK``, ``DP-8QAM``, ...), case
    insensitive and with dashes stripped.
    """
    normalized = name.strip().upper().replace("-", "")
    if normalized.startswith("DP"):
        normalized = normalized[2:]
    return {
        "QPSK": QPSK,
        "4QAM": QPSK,
        "8QAM": QAM8,
        "16QAM": QAM16,
        "64QAM": QAM64,
    }[normalized]()


# --------------------------------------------------------------------------- #
#  Default fibre parameters (SMF-28e style)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FibreParams:
    """Standard single-mode fibre constants used across the engine."""

    wavelength_nm: float = 1550.0
    alpha_db_km: float = 0.2
    dispersion_ps: float = 16.0  # [ps/(nm*km)]
    nonlinear_index_m2_w: float = 2.6e-20  # n2 [m^2/W]
    effective_area_um2: float = 80.0

    @property
    def beta2(self) -> float:
        """Group-velocity dispersion in [s^2/m]."""
        return beta2_from_D(self.wavelength_nm * 1e-9, self.dispersion_ps)

    @property
    def gamma(self) -> float:
        """Kerr coefficient in [1/(W*m)]."""
        return gamma_from_n2(
            self.wavelength_nm * 1e-9, self.nonlinear_index_m2_w, self.effective_area_um2 * 1e-12
        )
