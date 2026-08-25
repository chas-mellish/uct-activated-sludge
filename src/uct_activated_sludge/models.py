"""Data models for the UCT Activated Sludge simulation.

Typed dataclasses ported from the global variables declared in
VARSUNIT.PAS, with default values drawn from the Pascal initialization
blocks in KINUNIT.PAS, STCHUNIT.PAS, DIUNIT.PAS, and WATER.PAS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Constants mirrored from VARSUNIT.PAS
# ---------------------------------------------------------------------------

FIRST_SLOW: int = 1
LAST_SLOW: int = 7
FIRST_RAPID: int = 8
LAST_RAPID: int = 13
NO_PART: int = 7
TOTAL_COMPOUNDS: int = 14  # including DO
LAST_COMPOUND: int = 13
NO_PROCESSES: int = 14
NO_DI_VARS: int = 18

MAX_REAC: int = 12
MAX_REAC_P1: int = 13  # MaxReac + 1
MAX_N: int = 169  # LastCompound * (MaxReac + 1)
NO_DIURNAL_INTS: int = 12


# ---------------------------------------------------------------------------
# Helper to build 1-based arrays (index 0 is unused padding)
# ---------------------------------------------------------------------------

def _zeros_1based(size: int) -> npt.NDArray[np.float64]:
    """Return a float64 array of length *size + 1*, zero-filled.

    Index 0 is unused; valid indices are 1 .. size.
    """
    return np.zeros(size + 1, dtype=np.float64)


def _false_1based(size: int) -> npt.NDArray[np.bool_]:
    """Return a bool array of length *size + 1*, all False.

    Index 0 is unused; valid indices are 1 .. size.
    """
    return np.zeros(size + 1, dtype=np.bool_)


def _zeros_2d_1based(rows: int, cols: int) -> npt.NDArray[np.float64]:
    """Return a 2-D float64 array of shape (rows+1, cols+1), zero-filled.

    Row 0 and column 0 are unused padding so that Pascal-style 1-based
    indexing ``arr[k, i]`` works directly.
    """
    return np.zeros((rows + 1, cols + 1), dtype=np.float64)


# ===================================================================
# 1. KineticParams
# ===================================================================

@dataclass
class KineticParams:
    """Kinetic rate constants and Arrhenius temperature-correction factors.

    Default values are the 20 degC reference values from KINUNIT.PAS.
    The temperature-adjusted fields (without the ``20`` suffix) are
    initialised to the same 20 degC values; call the temperature-
    adjustment routine to update them for a given operating temperature.
    """

    # -- Heterotrophic 20 degC reference values --
    MuHatHetero20: float = 3.2
    Ks20: float = 5.0
    Bh20: float = 0.62
    Koh: float = 0.002
    NetaGrow: float = 0.33
    Kno: float = 0.10
    Kmp20: float = 1.35
    Ksp20: float = 0.027
    Kr20: float = 0.032
    Kna: float = 0.01
    Ka20: float = 0.17

    # -- Autotrophic 20 degC reference values --
    MuHatAuto20: float = 0.45
    Knh20: float = 1.0
    Koa: float = 0.002
    Ba20: float = 0.04

    # -- Arrhenius theta correction factors --
    ThetaMuHatH: float = 1.200
    ThetaKs: float = 1.000
    ThetaBh: float = 1.029
    ThetaKmp: float = 1.08
    ThetaKsp: float = 0.910
    ThetaKr: float = 1.029
    ThetaKa: float = 1.029
    ThetaMuHatA: float = 1.123
    ThetaKnh: float = 1.123
    ThetaBa: float = 1.029

    # -- Temperature-adjusted values (default = 20 degC values) --
    MuHatHetero: float = 3.2
    Ks: float = 5.0
    Bh: float = 0.62
    Kmp: float = 1.35
    Ksp: float = 0.027
    Kr: float = 0.032
    Ka: float = 0.17
    MuHatAuto: float = 0.45
    Knh: float = 1.0
    Ba: float = 0.04

    # -- Intermediate rate variables (computed at runtime) --
    MuHeteroS: float = 0.0
    MuHeteroP: float = 0.0
    NO3Limit: float = 0.0
    NH3Limit: float = 0.0
    NetaSol: float = 0.0


# ===================================================================
# 2. StoichiometricParams
# ===================================================================

@dataclass
class StoichiometricParams:
    """Yield and stoichiometric coefficients from STCHUNIT.PAS.

    Yh is expressed as COD-yield: 0.45 * 1.48 = 0.666.
    Ixb and Ixe are N-content fractions: 0.1 / 1.48 ~ 0.0676.
    """

    Yh: float = field(default_factory=lambda: 0.45 * 1.48)   # 0.666
    Fe: float = 0.08
    Ixb: float = field(default_factory=lambda: 0.1 / 1.48)   # ~0.0676
    Ixe: float = field(default_factory=lambda: 0.1 / 1.48)   # ~0.0676
    Ya: float = 0.15
    CODVSS: float = 1.48
    Fma: float = 1.00
    VSSTSS: float = 0.0  # declared in VARSUNIT, no init in STCHUNIT


# ===================================================================
# 3. WastewaterParams
# ===================================================================

@dataclass
class WastewaterParams:
    """Influent wastewater fractionation parameters from WATER.PAS.

    The ``settled`` flag selects the Settled preset when True; defaults
    shown here correspond to the **Raw** sewage preset.

    ``Sti`` and ``Nti`` are user-supplied influent concentrations
    (g COD m-3 and g N m-3 respectively).  Default values are set to
    typical raw sewage values for convenience.
    """

    # Influent concentrations (user-supplied; reasonable defaults)
    Sti: float = 500.0
    Nti: float = 50.0

    # Fractionation parameters -- Raw defaults from WATER.PAS
    Fbs: float = 0.20
    Fus: float = 0.05
    Fup: float = 0.13
    Fnaa: float = 0.75
    Fnox: float = 0.50
    Fnu: float = 0.03
    Fxbh: float = 0.0
    VSSTSS: float = 0.75

    # Influent alkalinity (g CaCO3 m-3 as milli-equivalents/L)
    Alki: float = 10.0

    # Whether settled-sewage fractionation is active
    settled: bool = False

    # Average values (tracking diurnal mean)
    StiAvg: float = 0.0
    NtiAvg: float = 0.0

    @classmethod
    def raw(cls, **kwargs: float) -> WastewaterParams:
        """Factory for Raw sewage fractionation (WATER.PAS ``Raw``)."""
        defaults = dict(
            Fbs=0.20, Fus=0.05, Fup=0.13,
            Fnaa=0.75, Fnox=0.50, Fnu=0.03,
            Fxbh=0.0, VSSTSS=0.75, settled=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def settled_sewage(cls, **kwargs: float) -> WastewaterParams:
        """Factory for Settled sewage fractionation (WATER.PAS ``Settled``)."""
        defaults = dict(
            Fbs=0.25, Fus=0.08, Fup=0.04,
            Fnaa=0.83, Fnox=0.50, Fnu=0.04,
            Fxbh=0.0, VSSTSS=0.83, settled=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)


# ===================================================================
# 4. PlantConfig
# ===================================================================

@dataclass
class PlantConfig:
    """Reactor topology and operating parameters.

    Arrays use 1-based indexing (index 0 is unused padding) to match the
    Pascal convention ``Vol[1..MaxReac]``, etc.
    """

    # Number of active reactors (1 .. MAX_REAC)
    LastReactor: int = 3

    # Reactor volumes (1-based, length MAX_REAC_P1+1)
    Vol: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )

    # Fractional feed distribution to each reactor
    FracFeed: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )

    # Dissolved-oxygen setpoint per reactor (g O2 m-3)
    DOConc: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )

    # Whether each reactor is aerated (True = aerated)
    ReactorAerated: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.ones(MAX_REAC_P1 + 1, dtype=np.bool_),
    )

    # Recycle flag arrays (1-based, 0/1 byte flags in Pascal)
    FlagRASIn: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.zeros(MAX_REAC_P1 + 1, dtype=np.int8),
    )
    FlagAIn: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.zeros(MAX_REAC_P1 + 1, dtype=np.int8),
    )
    FlagBIn: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.zeros(MAX_REAC_P1 + 1, dtype=np.int8),
    )
    FlagAOut: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.zeros(MAX_REAC_P1 + 1, dtype=np.int8),
    )
    FlagBOut: npt.NDArray[np.int8] = field(
        default_factory=lambda: np.zeros(MAX_REAC_P1 + 1, dtype=np.int8),
    )

    # Recycle source / destination reactor numbers
    ReactorAIn: int = 0
    ReactorAOut: int = 0
    ReactorBIn: int = 0
    ReactorBOut: int = 0

    # Flow parameters (volumetric rates, same units as Vol per day)
    FlowFeed: float = 0.0
    FlowFeedAvg: float = 0.0
    FlowRASrecycle: float = 0.0
    FlowArecycle: float = 0.0
    FlowBrecycle: float = 0.0
    FlowWaste: float = 0.0
    FlowWastePrevious: float = 0.0
    FlowWasteAvg: float = 0.0
    FlowToSettler: float = 0.0

    # Derived flow arrays (1-based)
    FlowFromPrevious: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )
    FlowInTotal: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )

    # Sludge retention time (days)
    Rs: float = 0.0

    # Operating temperature (degC)
    Temp: float = 20.0
    TempDiff: float = 0.0

    # Volume book-keeping
    VolumeTotal: float = 0.0
    VolumeUnaerated: float = 0.0
    CumVol: float = 0.0

    # Volume unit label (e.g. "ML", "m3")
    VolType: str = "ML"

    # Configuration status flags
    NoReactorsSet: bool = False
    VolumesSet: bool = False
    FeedDistribSet: bool = False
    RecyclesSet: bool = False
    ARecycleSet: bool = False
    BRecycleSet: bool = False
    FlowSet: bool = False
    SludgeAgeSet: bool = False
    TempSet: bool = False


# ===================================================================
# 5. ReactorState
# ===================================================================

@dataclass
class ReactorState:
    """Concentration and rate matrices for all reactors.

    All arrays use 1-based indexing to match the Pascal originals.

    ``C[k, i]`` is the concentration of compound *i* in reactor *k*.
    ``Rho[j]`` is the rate of process *j* (per-reactor, recomputed as needed).
    """

    # Concentration matrix: C[0..MaxReacP1, 1..TotalCompounds]
    C: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS),
    )

    # Steady-state reference concentrations (same shape as C)
    CSteady: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS),
    )

    # Influent compound vector C0[0..TotalCompounds]
    C0: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(TOTAL_COMPOUNDS),
    )

    # Process rate vector Rho[1..NoProcesses]  (per reactor, recomputed)
    Rho: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(NO_PROCESSES),
    )

    # Stoichiometric matrix Stoich[1..TotalCompounds, 1..NoProcesses]
    Stoich: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_2d_1based(TOTAL_COMPOUNDS, NO_PROCESSES),
    )

    # Oxygen uptake rates per reactor (1-based, 0..MaxReacP1)
    Oc: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )
    On: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )
    Ot: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )

    # Denitrification rate per reactor (1-based, 0..MaxReacP1)
    Denit: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )

    # Volatile suspended solids per reactor
    XvTot: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )

    # Air-supply switching functions per reactor
    AirOnHetero: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )
    AirOnAuto: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )
    AirOffHetero: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )
    AirOffAuto: npt.NDArray[np.float64] = field(
        default_factory=lambda: _zeros_1based(MAX_REAC_P1),
    )


# ===================================================================
# 6. IntegrationParams
# ===================================================================

@dataclass
class IntegrationParams:
    """Numerical integration control parameters from DIUNIT.PAS.

    ``Accuracy`` is the relative-error tolerance (fraction).
    ``Theta`` is the implicit/explicit weighting factor (0 = explicit,
    1 = fully implicit).
    ``DataIntHours`` is the data-interpolation interval in hours.
    """

    Accuracy: float = 0.50
    Theta: float = 0.70
    DataIntHours: float = 0.25  # 15 minutes, in hours

    # The following are computed at runtime by the integration routine
    # but are declared here so callers can inspect / override them.
    MaxDeltaTLarge: float = 0.0
    MaxDeltaTSmall: float = 0.0
    DeltaTlarge: float = 0.0
    DeltaTsmall: float = 0.0
    IntegIntDays: float = 0.0
    IntegTime: float = 0.0


# ===================================================================
# 7. SimulationState  (top-level container)
# ===================================================================

@dataclass
class SimulationState:
    """Aggregate state object holding every sub-model.

    This mirrors the flat global-variable namespace of the Pascal
    program, but groups related variables into typed sub-objects.
    """

    kinetics: KineticParams = field(default_factory=KineticParams)
    stoichiometry: StoichiometricParams = field(default_factory=StoichiometricParams)
    wastewater: WastewaterParams = field(default_factory=WastewaterParams)
    plant: PlantConfig = field(default_factory=PlantConfig)
    reactor: ReactorState = field(default_factory=ReactorState)
    integration: IntegrationParams = field(default_factory=IntegrationParams)

    # Cumulative / mean tracking variables from VARSUNIT.PAS
    CumFlow: float = 0.0
    CumCODLoad: float = 0.0
    CumTKNLoad: float = 0.0
    MeanFlow: float = 0.0
    MeanCOD: float = 0.0
    MeanTKN: float = 0.0

    # Simulation-status flags
    ConfigSpecified: bool = False
    OperationSpecified: bool = False
    SteadyStateSpecified: bool = False
    SteadyDataSpecified: bool = False
    DiurnalSpecified: bool = False
    DiurnalDataSpecified: bool = False
    WasteSpecified: bool = False
    NoChanges: bool = False
    GoAhead: bool = False
