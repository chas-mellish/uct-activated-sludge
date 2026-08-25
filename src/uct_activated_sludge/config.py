"""TOML-based configuration loading for the UCT Activated Sludge Model.

Provides helpers to load a TOML file, map its sections to the parameter
dataclasses defined in :mod:`uct_activated_sludge.models`, and load
diurnal input data from CSV or TOML.

Uses :mod:`tomllib` (Python 3.11+ stdlib) with a fallback to the
third-party ``tomli`` package for Python 3.10 compatibility.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

# Python 3.11+ ships tomllib in the stdlib.  For 3.10 we fall back to
# the compatible third-party package ``tomli``.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover — Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from uct_activated_sludge.models import (
    MAX_REAC_P1,
    IntegrationParams,
    KineticParams,
    PlantConfig,
    StoichiometricParams,
    WastewaterParams,
    _zeros_1based,
)

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

__all__ = [
    "load_config",
    "build_params_from_config",
    "load_diurnal_data",
]


# ===================================================================
# 1. load_config
# ===================================================================

def load_config(path: str) -> dict[str, Any]:
    """Load a TOML configuration file and return the parsed dict.

    Parameters
    ----------
    path : str
        Filesystem path to the ``.toml`` configuration file.

    Returns
    -------
    dict
        The parsed TOML content as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist, with a human-friendly message.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {resolved}\n"
            "Please check the path and try again."
        )
    with open(resolved, "rb") as fh:
        return tomllib.load(fh)


# ===================================================================
# 2. build_params_from_config
# ===================================================================

# Mapping from TOML section names to (DataclassType, flat-field-names,
# nested-subsection-mappings).  Nested subsections are keyed by their
# TOML sub-table name and map to the same flat dataclass fields.

_KINETIC_HETERO_FIELDS = {
    "MuHatHetero20", "Ks20", "Bh20", "Koh", "NetaGrow", "Kno",
    "Kmp20", "Ksp20", "Kr20", "Kna", "Ka20",
    # Temperature-adjusted twins (user may override directly)
    "MuHatHetero", "Ks", "Bh", "Kmp", "Ksp", "Kr", "Ka",
}

_KINETIC_AUTO_FIELDS = {
    "MuHatAuto20", "Knh20", "Koa", "Ba20",
    "MuHatAuto", "Knh", "Ba",
}

_KINETIC_ARRHENIUS_FIELDS = {
    "ThetaMuHatH", "ThetaKs", "ThetaBh", "ThetaKmp", "ThetaKsp",
    "ThetaKr", "ThetaKa", "ThetaMuHatA", "ThetaKnh", "ThetaBa",
}

# All valid KineticParams field names (used for top-level [kinetics])
_KINETIC_ALL_FIELDS = _KINETIC_HETERO_FIELDS | _KINETIC_AUTO_FIELDS | _KINETIC_ARRHENIUS_FIELDS


def _extract_kinetics(section: dict[str, Any]) -> dict[str, Any]:
    """Flatten a ``[kinetics]`` TOML section (with optional sub-tables).

    Supports:
    - ``[kinetics]`` with all fields flat
    - ``[kinetics.heterotrophs]`` / ``[kinetics.autotrophs]`` /
      ``[kinetics.arrhenius]`` nested subsections
    """
    kwargs: dict[str, Any] = {}

    # Nested subsection mappings
    nested_map = {
        "heterotrophs": _KINETIC_HETERO_FIELDS,
        "autotrophs": _KINETIC_AUTO_FIELDS,
        "arrhenius": _KINETIC_ARRHENIUS_FIELDS,
    }

    for key, value in section.items():
        if isinstance(value, dict):
            # It is a nested sub-table (e.g. [kinetics.heterotrophs])
            allowed = nested_map.get(key)
            if allowed is not None:
                for subkey, subval in value.items():
                    if subkey in allowed:
                        kwargs[subkey] = subval
        else:
            # Flat key directly under [kinetics]
            if key in _KINETIC_ALL_FIELDS:
                kwargs[key] = value

    return kwargs


def _build_plant_config(section: dict[str, Any]) -> PlantConfig:
    """Build a :class:`PlantConfig` from a ``[plant]`` TOML section.

    Handles array fields (Vol, FracFeed, DOConc, ReactorAerated, and
    the recycle flag arrays) by converting TOML lists into the 1-based
    NumPy arrays expected by the model.
    """
    pc = PlantConfig()

    # Array fields: TOML lists -> 1-based numpy arrays
    _array_float_fields = {"Vol", "FracFeed", "DOConc"}
    _array_bool_fields = {"ReactorAerated"}
    _array_int8_fields = {
        "FlagRASIn", "FlagAIn", "FlagBIn", "FlagAOut", "FlagBOut",
    }
    _flow_array_fields = {"FlowFromPrevious", "FlowInTotal"}

    for key, value in section.items():
        if key in _array_float_fields and isinstance(value, list):
            arr = _zeros_1based(MAX_REAC_P1)
            for i, v in enumerate(value, start=1):
                if i <= MAX_REAC_P1:
                    arr[i] = float(v)
            setattr(pc, key, arr)
        elif key in _array_bool_fields and isinstance(value, list):
            arr = np.ones(MAX_REAC_P1 + 1, dtype=np.bool_)
            for i, v in enumerate(value, start=1):
                if i <= MAX_REAC_P1:
                    arr[i] = bool(v)
            setattr(pc, key, arr)
        elif key in _array_int8_fields and isinstance(value, list):
            arr = np.zeros(MAX_REAC_P1 + 1, dtype=np.int8)
            for i, v in enumerate(value, start=1):
                if i <= MAX_REAC_P1:
                    arr[i] = int(v)
            setattr(pc, key, arr)
        elif key in _flow_array_fields and isinstance(value, list):
            arr = _zeros_1based(MAX_REAC_P1)
            for i, v in enumerate(value, start=1):
                if i <= MAX_REAC_P1:
                    arr[i] = float(v)
            setattr(pc, key, arr)
        elif hasattr(pc, key) and not isinstance(value, dict):
            # Scalar field — set directly with appropriate type
            current = getattr(pc, key)
            if isinstance(current, bool):
                setattr(pc, key, bool(value))
            elif isinstance(current, int):
                setattr(pc, key, int(value))
            elif isinstance(current, float):
                setattr(pc, key, float(value))
            elif isinstance(current, str):
                setattr(pc, key, str(value))

    if pc.VolumeTotal == 0.0 and pc.LastReactor > 0:
        pc.VolumeTotal = float(np.sum(pc.Vol[1:pc.LastReactor + 1]))

    return pc


def build_params_from_config(
    config: dict[str, Any],
) -> tuple[
    KineticParams,
    StoichiometricParams,
    WastewaterParams,
    PlantConfig,
    IntegrationParams,
]:
    """Construct all parameter dataclasses from a parsed TOML config dict.

    Missing keys use the dataclass defaults, so a completely empty dict
    produces the standard default parameter set.

    Parameters
    ----------
    config : dict
        Parsed TOML configuration (e.g. from :func:`load_config`).
        Expected top-level sections: ``kinetics``, ``stoichiometry``,
        ``wastewater``, ``plant``, ``integration``.

    Returns
    -------
    tuple
        ``(KineticParams, StoichiometricParams, WastewaterParams,
        PlantConfig, IntegrationParams)``
    """
    # -- KineticParams --
    kinetics_section = config.get("kinetics", {})
    kinetics_kwargs = _extract_kinetics(kinetics_section)
    kp = KineticParams(**kinetics_kwargs)

    # -- StoichiometricParams --
    stoich_section = config.get("stoichiometry", {})
    stoich_fields = {f.name for f in dataclasses.fields(StoichiometricParams)}
    stoich_kwargs = {
        k: v for k, v in stoich_section.items()
        if k in stoich_fields and not isinstance(v, dict)
    }
    sp = StoichiometricParams(**stoich_kwargs)

    # -- WastewaterParams --
    ww_section = config.get("wastewater", {})
    ww_fields = {f.name for f in dataclasses.fields(WastewaterParams)}
    ww_kwargs = {
        k: v for k, v in ww_section.items()
        if k in ww_fields and not isinstance(v, dict)
    }
    wp = WastewaterParams(**ww_kwargs)

    # -- PlantConfig --
    plant_section = config.get("plant", {})
    pc = _build_plant_config(plant_section)

    # -- IntegrationParams --
    integ_section = config.get("integration", {})
    integ_fields = {f.name for f in dataclasses.fields(IntegrationParams)}
    integ_kwargs = {
        k: v for k, v in integ_section.items()
        if k in integ_fields and not isinstance(v, dict)
    }
    ip = IntegrationParams(**integ_kwargs)

    return kp, sp, wp, pc, ip


# ===================================================================
# 3. load_diurnal_data
# ===================================================================

def load_diurnal_data(path: str) -> list[dict[str, float]]:
    """Load diurnal variation data from a CSV or TOML file.

    Expected format: 12 records each containing ``Time``, ``Flow``,
    ``COD``, and ``TKN`` fields.

    For CSV the first row must be a header with those column names.
    For TOML the data should be stored as a list of inline tables
    under a ``[[diurnal]]`` array-of-tables key.

    Parameters
    ----------
    path : str
        Filesystem path to the data file (``.csv`` or ``.toml``).

    Returns
    -------
    list[dict[str, float]]
        A list of dicts, each with keys ``Time``, ``Flow``, ``COD``,
        ``TKN``, and float values.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file format is not recognized or data is malformed.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Diurnal data file not found: {resolved}\n"
            "Please check the path and try again."
        )

    ext = resolved.suffix.lower()

    if ext == ".csv":
        return _load_diurnal_csv(resolved)
    elif ext == ".toml":
        return _load_diurnal_toml(resolved)
    else:
        raise ValueError(
            f"Unsupported diurnal data format '{ext}'. "
            "Use .csv or .toml."
        )


def _load_diurnal_csv(path: Path) -> list[dict[str, float]]:
    """Load diurnal data from a CSV file."""
    records: list[dict[str, float]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append({
                "Time": float(row["Time"]),
                "Flow": float(row["Flow"]),
                "COD": float(row["COD"]),
                "TKN": float(row["TKN"]),
            })
    return records


def _load_diurnal_toml(path: Path) -> list[dict[str, float]]:
    """Load diurnal data from a TOML file.

    Expects a ``[[diurnal]]`` array-of-tables or a ``diurnal`` key
    holding a list of inline tables.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    diurnal_list = data.get("diurnal")
    if diurnal_list is None:
        raise ValueError(
            f"TOML file {path} does not contain a 'diurnal' key. "
            "Expected [[diurnal]] array-of-tables with Time, Flow, COD, TKN."
        )
    if not isinstance(diurnal_list, list):
        raise ValueError(
            f"'diurnal' key in {path} must be a list (array-of-tables)."
        )

    records: list[dict[str, float]] = []
    for entry in diurnal_list:
        records.append({
            "Time": float(entry["Time"]),
            "Flow": float(entry["Flow"]),
            "COD": float(entry["COD"]),
            "TKN": float(entry["TKN"]),
        })
    return records
