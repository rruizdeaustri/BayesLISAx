"""Utilities for inspecting Sangria/LDC Galactic Binary catalogues.

The approximate SNR helper in this module is intentionally lightweight and is
only meant for diagnostics and frequency-window planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import h5py
import numpy as np

CATALOGUE_PATHS = {
    "dgb": "sky/dgb/cat",
    "igb": "sky/igb/cat",
    "vgb": "sky/vgb/cat",
}

EXPECTED_COLUMNS = (
    "Frequency",
    "Amplitude",
    "FrequencyDerivative",
    "EclipticLatitude",
    "EclipticLongitude",
    "Inclination",
    "InitialPhase",
    "Polarization",
)


@dataclass(frozen=True)
class Catalogue:
    """Normalized in-memory view of one Sangria source catalogue."""

    source_type: str
    path: str
    columns: tuple[str, ...]
    data: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        if not self.columns:
            return 0
        return int(len(self.data[self.columns[0]]))


def list_top_level_groups(handle: h5py.File) -> list[str]:
    """Return top-level HDF5 group/dataset names."""

    return sorted(str(key) for key in handle.keys())


def available_catalogue_paths(handle: h5py.File) -> dict[str, str]:
    """Return configured Sangria catalogue paths that are present in ``handle``."""

    return {
        source_type: path
        for source_type, path in CATALOGUE_PATHS.items()
        if path in handle
    }


def read_catalogue(handle: h5py.File, source_type: str, path: str) -> Catalogue:
    """Read a Sangria catalogue as column-oriented NumPy arrays.

    Sangria/LDC HDF5 files are encountered either as a group containing one
    dataset per catalogue column or as a single compound/table dataset.  This
    reader accepts both forms and returns a consistent dictionary.
    """

    node = handle[path]
    data: dict[str, np.ndarray] = {}

    if isinstance(node, h5py.Group):
        for name, child in node.items():
            if isinstance(child, h5py.Dataset):
                data[str(name)] = np.asarray(child[()])
    elif isinstance(node, h5py.Dataset):
        values = node[()]
        if values.dtype.names:
            for name in values.dtype.names:
                data[str(name)] = np.asarray(values[name])
        else:
            raise ValueError(f"Catalogue dataset {path!r} is not a compound table")
    else:
        raise TypeError(f"Unsupported HDF5 node type at {path!r}: {type(node)!r}")

    if not data:
        return Catalogue(source_type, path, tuple(), {})

    lengths = {name: len(np.atleast_1d(values)) for name, values in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Catalogue {path!r} has inconsistent column lengths: {lengths}")

    normalized = {name: np.atleast_1d(values) for name, values in data.items()}
    return Catalogue(source_type, path, tuple(sorted(normalized)), normalized)


def read_catalogues(h5_path: str) -> list[Catalogue]:
    """Read all configured source catalogues that are present in a Sangria file."""

    with h5py.File(h5_path, "r") as handle:
        return [
            read_catalogue(handle, source_type, path)
            for source_type, path in available_catalogue_paths(handle).items()
        ]


def validate_expected_columns(catalogue: Catalogue) -> list[str]:
    """Return expected Sangria columns missing from ``catalogue``."""

    present = set(catalogue.columns)
    return [name for name in EXPECTED_COLUMNS if name not in present]


def concatenate_catalogues(catalogues: Iterable[Catalogue]) -> dict[str, np.ndarray]:
    """Concatenate the columns needed for window diagnostics."""

    required = ("Frequency", "Amplitude", "EclipticLatitude", "Inclination")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in required}
    source_types: list[np.ndarray] = []

    for catalogue in catalogues:
        missing = [name for name in required if name not in catalogue.data]
        if missing:
            continue
        for name in required:
            chunks[name].append(np.asarray(catalogue.data[name], dtype=float))
        source_types.append(np.full(catalogue.size, catalogue.source_type, dtype=object))

    if not chunks["Frequency"]:
        return {name: np.array([], dtype=float) for name in required} | {
            "source_type": np.array([], dtype=object)
        }

    return {name: np.concatenate(parts) for name, parts in chunks.items()} | {
        "source_type": np.concatenate(source_types)
    }


def lisa_gb_approx_snr(
    frequency_hz: np.ndarray,
    amplitude: np.ndarray,
    tobs_s: float,
    ecliptic_latitude: np.ndarray | None = None,
    inclination: np.ndarray | None = None,
) -> np.ndarray:
    """Approximate sky-averaged monochromatic Galactic Binary LISA SNR.

    This simple diagnostic follows the common planning estimate
    ``SNR^2 ~= A^2 * Tobs * <F^2> / S_n(f)`` with a Robson-Cornish-Liu style
    analytic LISA noise curve and a mild inclination response factor.  It is
    not used by the production BayesLISAx likelihood.
    """

    f = np.asarray(frequency_hz, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    safe_f = np.clip(f, 1.0e-5, None)

    arm_length = 2.5e9
    f_star = 19.09e-3
    p_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / safe_f) ** 4)
    p_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / safe_f) ** 2) * (1.0 + (safe_f / 8.0e-3) ** 4)
    noise = (10.0 / (3.0 * arm_length**2)) * (p_oms + 4.0 * p_acc / (2.0 * np.pi * safe_f) ** 4)
    noise *= 1.0 + 0.6 * (safe_f / f_star) ** 2

    response = np.full_like(safe_f, 3.0 / 20.0)
    if inclination is not None:
        cos_i = np.cos(np.asarray(inclination, dtype=float))
        response *= ((1.0 + cos_i**2) ** 2 / 4.0 + cos_i**2) / 2.0
    if ecliptic_latitude is not None:
        beta = np.asarray(ecliptic_latitude, dtype=float)
        response *= 0.75 + 0.25 * np.cos(beta) ** 2

    snr2 = amp**2 * float(tobs_s) * response / noise
    return np.sqrt(np.maximum(snr2, 0.0))
