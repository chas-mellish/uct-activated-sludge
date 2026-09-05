"""Shared fixtures for the UCT Activated Sludge Model test suite."""

from pathlib import Path

import pytest

from uct_activated_sludge.config import (
    build_params_from_config,
    load_config,
    load_diurnal_data,
)
from uct_activated_sludge.cli import _build_parser
from uct_activated_sludge.models import (
    KineticParams,
    PlantConfig,
    StoichiometricParams,
    WastewaterParams,
)
from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def default_kinetic_params():
    """Default kinetic parameters at 20 degC reference values."""
    return KineticParams()


@pytest.fixture
def default_stoich_params():
    """Default stoichiometric/yield parameters."""
    return StoichiometricParams()


@pytest.fixture
def default_ww_params():
    """Default raw wastewater parameters."""
    return WastewaterParams()


@pytest.fixture
def default_plant_config():
    """A typical 3-reactor UCT plant configuration.

    Anaerobic (reactor 1) -> Anoxic (reactor 2) -> Aerobic (reactor 3)
    with RAS recycled to reactor 1 and A-recycle from reactor 3 to reactor 2.
    """
    pc = PlantConfig()
    pc.LastReactor = 3

    # Reactor volumes (ML -- consistent with FlowFeed in ML/d)
    pc.Vol[1] = 1.5   # anaerobic
    pc.Vol[2] = 3.0   # anoxic
    pc.Vol[3] = 6.0   # aerobic

    # All feed to reactor 1
    pc.FracFeed[1] = 1.0

    # Dissolved oxygen setpoints
    pc.DOConc[1] = 0.0   # anaerobic
    pc.DOConc[2] = 0.0   # anoxic
    pc.DOConc[3] = 2.0   # aerobic

    # Aeration flags
    pc.ReactorAerated[1] = False
    pc.ReactorAerated[2] = False
    pc.ReactorAerated[3] = True

    # Flow rates (ML/d)
    pc.FlowFeed = 25.0
    pc.FlowRASrecycle = 25.0
    pc.FlowArecycle = 75.0
    pc.FlowBrecycle = 0.0

    # Recycle routing: RAS goes to reactor 1
    pc.FlagRASIn[1] = 1

    # A-recycle: out of reactor 3, into reactor 2
    pc.FlagAIn[2] = 1
    pc.FlagAOut[3] = 1
    pc.ReactorAIn = 2
    pc.ReactorAOut = 3

    # Sludge age and temperature
    pc.Rs = 20.0
    pc.Temp = 20.0

    # Total volume (ML)
    pc.VolumeTotal = 10.5

    return pc


@pytest.fixture
def default_stoich_matrix(default_stoich_params):
    """Build the stoichiometric matrix from default parameters."""
    return build_stoichiometric_matrix(default_stoich_params)


@pytest.fixture
def cli_parser():
    """Build the CLI argument parser."""
    return _build_parser()


@pytest.fixture
def diurnal_pattern_data():
    """Load the example diurnal CSV pattern."""
    return load_diurnal_data(str(_EXAMPLES_DIR / "diurnal_pattern.csv"))


@pytest.fixture
def default_plant_toml_config():
    """Load the default_plant.toml example as a raw config dict."""
    return load_config(str(_EXAMPLES_DIR / "default_plant.toml"))


@pytest.fixture
def default_plant_params(default_plant_toml_config):
    """Build parameter objects from the default_plant.toml config."""
    return build_params_from_config(default_plant_toml_config)
