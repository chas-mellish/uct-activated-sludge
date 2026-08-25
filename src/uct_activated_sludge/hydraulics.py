"""Flow division, wastage, and hydraulic calculations.

Ported from the original Pascal source files:
  - FLOWSS.PAS   -- FlowDivisionSS  (steady-state flow division)
  - FLOWDI.PAS   -- FlowDivision    (dynamic flow division with dilution rates)
  - WASTE.PAS    -- WastageAndFlows  (iterative wastage rate via tracer)
  - SETWASTE.PAS -- SetWaste         (distribute wastage across diurnal intervals)

All arrays use 1-based indexing (index 0 is unused padding) to match
the Pascal original and simplify cross-validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uct_activated_sludge.constants import (
    MAX_REAC_P1,
    NO_PART,
    TOTAL_COMPOUNDS,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from uct_activated_sludge.models import PlantConfig

__all__ = [
    "flow_division_ss",
    "flow_division_dynamic",
    "wastage_and_flows",
    "set_waste",
]


# ---------------------------------------------------------------------------
# 1. flow_division_ss  --  ported from FLOWSS.PAS  (Procedure FlowDivisionSS)
# ---------------------------------------------------------------------------

def flow_division_ss(
    *,
    last_reactor: int,
    flow_feed: float,
    flow_ras_recycle: float,
    flow_a_recycle: float,
    flow_b_recycle: float,
    flow_waste: float,
    frac_feed: npt.NDArray[np.float64],
    flag_ras_in: npt.NDArray[np.int8],
    flag_a_in: npt.NDArray[np.int8],
    flag_b_in: npt.NDArray[np.int8],
    flag_a_out: npt.NDArray[np.int8],
    flag_b_out: npt.NDArray[np.int8],
) -> dict:
    """Steady-state flow distribution across reactors.

    Computes *FlowInTotal[k]* and *FlowFromPrevious[k]* for every reactor
    ``k`` in ``1 .. last_reactor``, plus the settling-tank pseudo-reactor
    at index ``last_reactor + 1``.

    Parameters
    ----------
    last_reactor : int
        Number of active reactors (1-based).
    flow_feed, flow_ras_recycle, flow_a_recycle, flow_b_recycle, flow_waste : float
        Volumetric flow rates (consistent units, e.g. ML/d).
    frac_feed : ndarray, shape (MAX_REAC_P1+1,)
        Feed-fraction to each reactor (1-based).
    flag_ras_in, flag_a_in, flag_b_in : ndarray
        Recycle-in flags (0 or 1) for each reactor.
    flag_a_out, flag_b_out : ndarray
        Recycle-out flags (0 or 1) for each reactor.

    Returns
    -------
    dict with keys:

    - ``"FlowFromPrevious"`` : ndarray -- flow entering from the upstream
      reactor (1-based, index 0 unused).
    - ``"FlowInTotal"`` : ndarray -- total flow into each reactor.
    """
    flow_from_previous = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    flow_in_total = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)

    flow_from_previous[1] = 0.0

    for k in range(1, last_reactor + 1):
        flow_in_total[k] = (
            flow_from_previous[k]
            + flow_ras_recycle * flag_ras_in[k]
            + flow_a_recycle * flag_a_in[k]
            + flow_b_recycle * flag_b_in[k]
            + flow_feed * frac_feed[k]
        )
        flow_from_previous[k + 1] = (
            flow_in_total[k]
            - flow_a_recycle * flag_a_out[k]
            - flow_b_recycle * flag_b_out[k]
        )

    # Settling tank (pseudo-reactor at LastReactor+1)
    flow_from_previous[last_reactor + 1] = (
        flow_feed + flow_ras_recycle - flow_waste
    )
    flow_in_total[last_reactor + 1] = flow_ras_recycle

    return {
        "FlowFromPrevious": flow_from_previous,
        "FlowInTotal": flow_in_total,
    }


def flow_division_ss_inplace(plant: PlantConfig) -> None:
    """In-place variant that updates *plant.FlowFromPrevious* and
    *plant.FlowInTotal* directly on a :class:`PlantConfig` instance.

    This mirrors the Pascal procedure which mutated module-level arrays.
    """
    result = flow_division_ss(
        last_reactor=plant.LastReactor,
        flow_feed=plant.FlowFeed,
        flow_ras_recycle=plant.FlowRASrecycle,
        flow_a_recycle=plant.FlowArecycle,
        flow_b_recycle=plant.FlowBrecycle,
        flow_waste=plant.FlowWaste,
        frac_feed=plant.FracFeed,
        flag_ras_in=plant.FlagRASIn,
        flag_a_in=plant.FlagAIn,
        flag_b_in=plant.FlagBIn,
        flag_a_out=plant.FlagAOut,
        flag_b_out=plant.FlagBOut,
    )
    plant.FlowFromPrevious[:] = result["FlowFromPrevious"]
    plant.FlowInTotal[:] = result["FlowInTotal"]


# ---------------------------------------------------------------------------
# 2. flow_division_dynamic  --  ported from FLOWDI.PAS  (Procedure FlowDivision)
# ---------------------------------------------------------------------------

def flow_division_dynamic(
    *,
    last_reactor: int,
    flow_feed: float,
    flow_ras_recycle: float,
    flow_a_recycle: float,
    flow_b_recycle: float,
    wastage_rate: float,
    frac_feed: npt.NDArray[np.float64],
    vol: npt.NDArray[np.float64],
    flag_ras_in: npt.NDArray[np.int8],
    flag_a_in: npt.NDArray[np.int8],
    flag_b_in: npt.NDArray[np.int8],
    flag_a_out: npt.NDArray[np.int8],
    flag_b_out: npt.NDArray[np.int8],
) -> dict:
    """Dynamic flow distribution with dilution rates for the integrator.

    In addition to *FlowInTotal* and *FlowFromPrevious* (same as the
    steady-state case), this routine computes per-reactor dilution rates:

    - ``DhOut[k]``           -- total outflow dilution rate
    - ``DhFromPrevious[k]``  -- dilution rate from the upstream reactor
    - ``DhFeed[k]``          -- dilution rate of the feed stream
    - ``DhRAS[k, i]``        -- dilution rate of the RAS recycle (compound-
      dependent: particulate vs. soluble)
    - ``DhArecycle[k]``      -- dilution rate of the A-recycle
    - ``DhBrecycle[k]``      -- dilution rate of the B-recycle

    Parameters
    ----------
    last_reactor : int
        Number of active reactors (1-based).
    flow_feed, flow_ras_recycle, flow_a_recycle, flow_b_recycle : float
        Volumetric flow rates.
    wastage_rate : float
        Current wastage flow rate for the diurnal interval.
    frac_feed : ndarray
        Feed-fraction to each reactor (1-based).
    vol : ndarray
        Reactor volumes (1-based).
    flag_ras_in, flag_a_in, flag_b_in : ndarray
        Recycle-in flags (0 or 1) for each reactor.
    flag_a_out, flag_b_out : ndarray
        Recycle-out flags (0 or 1) for each reactor.

    Returns
    -------
    dict with keys:

    - ``"FlowInTotal"`` : ndarray
    - ``"FlowFromPrevious"`` : ndarray
    - ``"DhOut"`` : ndarray  -- total outflow dilution rate per reactor
    - ``"DhFromPrevious"`` : ndarray
    - ``"DhFeed"`` : ndarray
    - ``"DhRAS"`` : ndarray, shape (MAX_REAC_P1+1, TOTAL_COMPOUNDS)
      -- per-compound RAS dilution rate
    - ``"DhArecycle"`` : ndarray
    - ``"DhBrecycle"`` : ndarray
    """
    flow_from_previous = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    flow_in_total = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    dh_out = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    dh_from_previous = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    dh_feed = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    dh_a_recycle = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    dh_b_recycle = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    # DhRAS is compound-dependent: particulate compounds use a different
    # flow (underflow) than soluble compounds (RAS recycle flow).
    # Shape: [0..MAX_REAC_P1, 0..TOTAL_COMPOUNDS]  (1-based on both axes)
    dh_ras = np.zeros((MAX_REAC_P1 + 1, TOTAL_COMPOUNDS + 1), dtype=np.float64)

    flow_from_previous[1] = 0.0

    for k in range(1, last_reactor + 1):
        flow_in_total[k] = (
            flow_from_previous[k]
            + flow_ras_recycle * flag_ras_in[k]
            + flow_a_recycle * flag_a_in[k]
            + flow_b_recycle * flag_b_in[k]
            + flow_feed * frac_feed[k]
        )
        flow_from_previous[k + 1] = (
            flow_in_total[k]
            - flow_a_recycle * flag_a_out[k]
            - flow_b_recycle * flag_b_out[k]
        )

        v = vol[k]
        dh_out[k] = flow_in_total[k] / v
        dh_from_previous[k] = flow_from_previous[k] / v
        dh_a_recycle[k] = flow_a_recycle * flag_a_in[k] / v
        dh_b_recycle[k] = flow_b_recycle * flag_b_in[k] / v
        dh_feed[k] = flow_feed * frac_feed[k] / v

        # RAS dilution rates differ for particulate vs. soluble compounds.
        # Particulate compounds travel with the underflow:
        #   Q_underflow = FlowFeed + FlowRASrecycle - WastageRate
        # Soluble compounds travel with the RAS recycle flow.
        dh_ras_part = (
            (flow_feed + flow_ras_recycle - wastage_rate)
            * flag_ras_in[k] / v
        )
        dh_ras_sol = flow_ras_recycle * flag_ras_in[k] / v

        if flow_ras_recycle == 0.0:
            dh_ras_part = 0.0
            dh_ras_sol = 0.0

        # Pascal: for i := 1 to (TotalCompounds-1) do
        #           if i > NoPart then DhRAS[k,i] := DhRASsol
        #                         else DhRAS[k,i] := DhRASpart;
        for i in range(1, TOTAL_COMPOUNDS):  # 1 .. TotalCompounds-1
            if i > NO_PART:
                dh_ras[k, i] = dh_ras_sol
            else:
                dh_ras[k, i] = dh_ras_part

    return {
        "FlowInTotal": flow_in_total,
        "FlowFromPrevious": flow_from_previous,
        "DhOut": dh_out,
        "DhFromPrevious": dh_from_previous,
        "DhFeed": dh_feed,
        "DhRAS": dh_ras,
        "DhArecycle": dh_a_recycle,
        "DhBrecycle": dh_b_recycle,
    }


# ---------------------------------------------------------------------------
# 3. wastage_and_flows  --  ported from WASTE.PAS  (Procedure WastageAndFlows)
# ---------------------------------------------------------------------------
#
# NOTE: This procedure calls the Newton solver (solver.py) internally with
# Compounds=1 to solve the tracer mass-balance.  If solver.py is not yet
# implemented, the import below will succeed (the module file exists) but
# calling ``wastage_and_flows`` will raise at runtime.  This is intentional:
# the function signature and logic are correct; only the solver dependency
# needs to be fulfilled.
# ---------------------------------------------------------------------------

def _seed_waste(
    *,
    last_reactor: int,
    flow_feed: float,
    flow_ras_recycle: float,
    volume_total: float,
    rs: float,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Compute the initial wastage estimate and tracer seed concentrations.

    Ported from the nested ``SeedWaste`` procedure in WASTE.PAS.

    Parameters
    ----------
    last_reactor : int
        Number of active reactors.
    flow_feed : float
        Influent flow rate.
    flow_ras_recycle : float
        RAS recycle flow rate.
    volume_total : float
        Total reactor volume.
    rs : float
        Sludge retention time (days).

    Returns
    -------
    flow_waste : float
        Initial wastage-rate estimate.
    tracer : ndarray, shape (MAX_REAC_P1 + 1,)
        Seed tracer concentrations (1-based; index 0 is the feed at 0 mg/L).
    """
    tracer_conc_in = 100.0
    flow_waste = volume_total / rs  # initial estimate

    tracer = np.zeros(MAX_REAC_P1 + 1, dtype=np.float64)
    tracer[0] = 0.0
    for k in range(1, last_reactor + 1):
        tracer[k] = flow_feed * tracer_conc_in * rs / volume_total

    # Settling-tank pseudo-reactor
    if flow_ras_recycle == 0.0:
        tracer[last_reactor + 1] = 0.0
    else:
        tracer[last_reactor + 1] = (
            (flow_feed + flow_ras_recycle - flow_waste)
            / flow_ras_recycle
            * tracer[last_reactor]
        )

    return flow_waste, tracer


def wastage_and_flows(
    *,
    plant: PlantConfig,
    C: npt.NDArray[np.float64],
    newton_fn,
    scale_values_fn,
    max_iterations: int = 200,
) -> float:
    """Iterative wastage flow-rate calculation using a tracer mass balance.

    Ported from ``Procedure WastageAndFlows`` in WASTE.PAS.

    The algorithm:
    1. Seed the tracer concentrations and an initial wastage estimate.
    2. Repeat:
       a. Compute steady-state flow division (``flow_division_ss``).
       b. Copy tracer concentrations into ``C[k, 1]`` for all reactors.
       c. Call ``scale_values_fn(1)`` then ``newton_fn(1)`` to solve the
          single-compound (tracer) mass balance.
       d. Recover updated tracer concentrations from ``C[k, 1]``.
       e. Compute total tracer mass across all reactors.
       f. Update wastage flow:
          ``FlowWaste = (MassTracer / Rs) / Tracer[LastReactor]``.
    3. Until ``|FlowWaste - FlowWastePrevious| / FlowWaste < 0.0001``.

    Parameters
    ----------
    plant : PlantConfig
        Must have all flow and topology fields populated.  The function
        updates ``plant.FlowWaste``, ``plant.FlowFromPrevious``, and
        ``plant.FlowInTotal`` in place.
    C : ndarray, shape (MAX_REAC_P1+1, TOTAL_COMPOUNDS+1)
        Concentration matrix (modified in place for compound index 1).
    newton_fn : callable
        ``newton_fn(compounds)`` -- the Newton solver accepting the number
        of compounds to solve for.  For the tracer case, called as
        ``newton_fn(1)``.
    scale_values_fn : callable
        ``scale_values_fn(compounds)`` -- scaling routine called before
        Newton.  Called as ``scale_values_fn(1)``.
    max_iterations : int
        Safety cap on iterations (default 200).

    Returns
    -------
    float
        Converged wastage flow rate (``FlowWaste``).
    """
    flow_waste, tracer = _seed_waste(
        last_reactor=plant.LastReactor,
        flow_feed=plant.FlowFeed,
        flow_ras_recycle=plant.FlowRASrecycle,
        volume_total=plant.VolumeTotal,
        rs=plant.Rs,
    )
    plant.FlowWaste = flow_waste

    iteration = 0
    while True:
        flow_waste_previous = plant.FlowWaste

        # (a) Steady-state flow division (updates plant in place)
        flow_division_ss_inplace(plant)

        # (b) Copy tracer into C[k, 1]
        for k in range(1, plant.LastReactor + 2):  # 1 .. LastReactor+1
            C[k, 1] = tracer[k]

        # (c) Scale and solve (single compound)
        scale_values_fn(1)
        newton_fn(1)

        # (d) Recover updated tracer from C
        for k in range(1, plant.LastReactor + 2):
            tracer[k] = C[k, 1]

        # (e) Compute total tracer mass
        mass_tracer = 0.0
        for k in range(1, plant.LastReactor + 1):  # 1 .. LastReactor
            mass_tracer += tracer[k] * plant.Vol[k]

        # (f) Update wastage flow rate
        if tracer[plant.LastReactor] == 0.0:
            break
        plant.FlowWaste = (mass_tracer / plant.Rs) / tracer[plant.LastReactor]

        iteration += 1

        # Convergence check
        if plant.FlowWaste != 0.0:
            if abs(plant.FlowWaste - flow_waste_previous) / abs(plant.FlowWaste) < 0.0001:
                break

        if iteration >= max_iterations:
            break

    plant.FlowWasteAvg = plant.FlowWaste
    return plant.FlowWaste


# ---------------------------------------------------------------------------
# 4. set_waste  --  ported from SETWASTE.PAS  (Procedure SetWaste)
# ---------------------------------------------------------------------------

def set_waste(
    *,
    no_diurnal_ints: int,
    dynamic_flow: npt.NDArray[np.float64],
    flow_waste_avg: float,
) -> dict:
    """Distribute average daily wastage across diurnal time intervals.

    Wastage occurs only during intervals where the dynamic flow exceeds
    the average wastage rate.  The total daily wastage mass is preserved
    by distributing it evenly across the "wastage-on" intervals.

    Ported from ``Procedure SetWaste`` in SETWASTE.PAS.

    Parameters
    ----------
    no_diurnal_ints : int
        Number of diurnal intervals (typically 12).
    dynamic_flow : ndarray, shape (no_diurnal_ints+1,) or larger
        Diurnal flow pattern (1-based; index 0 unused).
        ``dynamic_flow[i]`` is the flow during interval *i*.
    flow_waste_avg : float
        Average daily wastage flow rate.

    Returns
    -------
    dict with keys:

    - ``"WastageOn"`` : ndarray of bool, shape (no_diurnal_ints+1,)
      -- whether wastage occurs in each interval (1-based).
    - ``"FlowWaste"`` : ndarray of float, shape (no_diurnal_ints+1,)
      -- wastage flow rate per interval (1-based; 0 when not wasting).
    - ``"NoWasteInts"`` : int
      -- number of intervals during which wastage is active.
    """
    wastage_on = np.zeros(no_diurnal_ints + 1, dtype=np.bool_)
    flow_waste = np.zeros(no_diurnal_ints + 1, dtype=np.float64)

    # Count intervals where the dynamic flow exceeds the average wastage.
    no_waste_ints = no_diurnal_ints
    for i in range(1, no_diurnal_ints + 1):
        if dynamic_flow[i] > flow_waste_avg:
            wastage_on[i] = True
        else:
            no_waste_ints -= 1
            wastage_on[i] = False

    if no_waste_ints == 0:
        raise ValueError(
            "All diurnal intervals have flow at or below the wastage average — "
            "zero active intervals for wastage distribution."
        )

    for i in range(1, no_diurnal_ints + 1):
        if wastage_on[i]:
            flow_waste[i] = flow_waste_avg * no_diurnal_ints / no_waste_ints
        else:
            flow_waste[i] = 0.0

    return {
        "WastageOn": wastage_on,
        "FlowWaste": flow_waste,
        "NoWasteInts": no_waste_ints,
    }
