"""Diurnal dynamic simulation for the UCT Activated Sludge Model.

Ported from the original Pascal source files:
  - DIURNAL.PAS   -- Diurnal procedure (cycle-based simulation loop)
  - DIUNIT.PAS    -- parent unit (variable declarations, initial values)
  - DIDATA.PAS    -- DiurnalData procedure (input data handling)
  - DIRSLTS.PAS   -- StoreResponse procedure (time-series storage)
  - MODWASTE.PAS  -- ModifyWaste procedure (adjust wastage during diurnal)
  - SETAVG.PAS    -- SetAvgInputs procedure (restore average conditions)

The diurnal simulation cycles over 12 two-hourly influent data intervals,
integrating the ASM1 model with adaptive time-stepping until the solution
reaches a periodic steady state (convergence within 0.7% for Xbh and Sbp
between start and end of a 24-hour cycle).

All arrays use 1-based indexing (index 0 is unused padding) to match the
Pascal original and simplify cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from uct_activated_sludge.constants import (
    ASM1Component as Comp,
    NO_DI_VARS,
    NO_DIURNAL_INTS,
    TOTAL_COMPOUNDS,
)
from uct_activated_sludge.hydraulics import flow_division_dynamic, set_waste
from uct_activated_sludge.influent import fractionate_influent
from uct_activated_sludge.integrator import integrate, integration_parameters
from uct_activated_sludge.kinetics import air_supply, utilization_rates
from uct_activated_sludge.models import (
    IntegrationParams,
    KineticParams,
    MAX_REAC_P1,
    PlantConfig,
    StoichiometricParams,
    WastewaterParams,
)

__all__ = [
    "run_diurnal",
    "store_response",
    "default_diurnal_data",
    "modify_waste",
    "DiurnalData",
]


# ---------------------------------------------------------------------------
# DiurnalData: typed container for a single 2-hourly influent record
# ---------------------------------------------------------------------------

@dataclass
class DiurnalData:
    """A single 2-hourly influent data record for the diurnal simulation.

    Attributes
    ----------
    Time : float
        Time of day in hours (0, 2, 4, ..., 22).
    Flow : float
        Influent flow rate during this interval (same units as PlantConfig
        flow fields, e.g. ML/d).
    COD : float
        Influent total COD during this interval (g COD m-3).
    TKN : float
        Influent TKN during this interval (g N m-3).
    """

    Time: float = 0.0
    Flow: float = 0.0
    COD: float = 0.0
    TKN: float = 0.0


# ---------------------------------------------------------------------------
# 1.  store_response  --  ported from DIURNAL.PAS StoreResponse
# ---------------------------------------------------------------------------

def store_response(
    C: npt.NDArray[np.float64],
    Oc: npt.NDArray[np.float64],
    On: npt.NDArray[np.float64],
    Ot: npt.NDArray[np.float64],
    stoich_params: StoichiometricParams,
    ww_params: WastewaterParams,
    plant_config: PlantConfig,
    response: list[npt.NDArray[np.float64] | None],
    data_no: int,
) -> None:
    """Store time-series response for all reactors at a given data index.

    Writes 18 output variables per reactor into the *response* arrays
    at column *data_no*:
      - Indices 1..13  : the 13 ASM1 compound concentrations
      - Index 14       : OURc (carbonaceous OUR, g O2 m-3 h-1)
      - Index 15       : OURn (nitrogenous OUR, g O2 m-3 h-1)
      - Index 16       : OURt (total OUR, g O2 m-3 h-1)
      - Index 17       : VSS  (volatile suspended solids, g VSS m-3)
      - Index 18       : TKN  (total Kjeldahl nitrogen, g N m-3)

    Ported from ``Procedure StoreResponse(DataNo : integer)`` in
    DIURNAL.PAS:

    .. code-block:: pascal

        for k := 1 to LastReactor do
          begin
            for i := 1 to (TotalCompounds - 1) do
              Response[k]^[i, DataNo] := C[k, i];
            Response[k]^[14, DataNo] := Oc[k];
            Response[k]^[15, DataNo] := On[k];
            Response[k]^[16, DataNo] := Ot[k];
            Response[k]^[17, DataNo] := 0.0;
            for j := 1 to 6 do
              Response[k]^[17, DataNo] := Response[k]^[17, DataNo]
                                          + C[k, j] / CODVSS;
            Response[k]^[18, DataNo] := C[k, 9] + C[k, 10] + Fnu * Nti;
          end;

    Parameters
    ----------
    C : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Concentration matrix (1-based).
    Oc, On, Ot : ndarray, shape (MaxReacP1+1,)
        Oxygen-uptake rates (1-based, per reactor).
    stoich_params : StoichiometricParams
        Needed for ``CODVSS``.
    ww_params : WastewaterParams
        Needed for ``Fnu`` and ``Nti``.
    plant_config : PlantConfig
        Needed for ``LastReactor``.
    response : list of ndarray or None
        1-based list.  ``response[k]`` is an ndarray of shape
        ``(NO_DI_VARS+1, data_per_day+1)`` for reactor *k*.
        Index 0 is None (unused).
    data_no : int
        Column index into the response arrays (0 = start of cycle).
    """
    CODVSS = stoich_params.CODVSS
    Fnu = ww_params.Fnu
    Nti = ww_params.Nti

    for k in range(1, plant_config.LastReactor + 1):
        resp_k = response[k]
        # 13 compound concentrations
        for i in range(1, TOTAL_COMPOUNDS):  # 1 .. TotalCompounds - 1 = 13
            resp_k[i, data_no] = C[k, i]

        # Oxygen uptake rates
        resp_k[14, data_no] = Oc[k]
        resp_k[15, data_no] = On[k]
        resp_k[16, data_no] = Ot[k]

        # VSS: sum of C[k, 1..6] / CODVSS
        vss = 0.0
        for j in range(1, 7):  # 1 .. 6
            vss += C[k, j] / CODVSS
        resp_k[17, data_no] = vss

        # TKN: Snh + Snd + Fnu * Nti
        resp_k[18, data_no] = C[k, Comp.Snh] + C[k, Comp.Snd] + Fnu * Nti


# ---------------------------------------------------------------------------
# 2.  default_diurnal_data  --  from DIDATA.PAS
# ---------------------------------------------------------------------------

def default_diurnal_data(
    flow: float = 15.0,
    cod: float = 500.0,
    tkn: float = 50.0,
) -> list[DiurnalData]:
    """Return a default (flat/constant) diurnal data set of 12 records.

    The Pascal DIDATA.PAS does not define hard-coded default values;
    all data must be entered by the user (keyboard or disk).  This
    function provides a convenience default: 12 two-hourly records
    with constant flow, COD, and TKN matching the ``WastewaterParams``
    defaults.

    Parameters
    ----------
    flow : float
        Constant influent flow rate for all intervals (default 15.0,
        matching a typical raw sewage design flow).
    cod : float
        Constant influent COD for all intervals (default 500.0 g COD m-3).
    tkn : float
        Constant influent TKN for all intervals (default 50.0 g N m-3).

    Returns
    -------
    list of DiurnalData
        12 records at times 0, 2, 4, ..., 22 hours.
    """
    records: list[DiurnalData] = []
    for i in range(NO_DIURNAL_INTS):
        records.append(DiurnalData(
            Time=float(i * 2),
            Flow=flow,
            COD=cod,
            TKN=tkn,
        ))
    return records


# ---------------------------------------------------------------------------
# 3.  modify_waste  --  ported from MODWASTE.PAS
# ---------------------------------------------------------------------------

def modify_waste(
    *,
    no_diurnal_ints: int,
    dynamic_flow: npt.NDArray[np.float64],
    flow_waste_avg: float,
    wastage_on: npt.NDArray[np.bool_],
    flow_waste: npt.NDArray[np.float64],
    no_waste_ints: int,
    record_no: int,
) -> dict[str, Any]:
    """Toggle wastage on/off for a specific diurnal interval.

    Ported from ``Procedure ModifyWastage`` in MODWASTE.PAS.

    The wastage at interval *record_no* is toggled on or off (if the
    dynamic flow in that interval exceeds the threshold).  The wastage
    flow is then redistributed evenly across all active intervals.

    Parameters
    ----------
    no_diurnal_ints : int
        Number of diurnal intervals (typically 12).
    dynamic_flow : ndarray, shape (no_diurnal_ints+1,)
        Diurnal flow pattern (1-based).
    flow_waste_avg : float
        Average daily wastage flow rate.
    wastage_on : ndarray of bool, shape (no_diurnal_ints+1,)
        Current wastage-on flags (1-based).  Modified in-place.
    flow_waste : ndarray of float, shape (no_diurnal_ints+1,)
        Current wastage flow per interval (1-based).  Modified in-place.
    no_waste_ints : int
        Current number of active wastage intervals.
    record_no : int
        1-based interval index to toggle.

    Returns
    -------
    dict with keys:

    - ``"WastageOn"``   : ndarray of bool (same object, modified)
    - ``"FlowWaste"``   : ndarray of float (same object, modified)
    - ``"NoWasteInts"`` : int (updated count)
    - ``"Changed"``     : bool (whether the toggle was applied)
    """
    changed = False

    # Pascal: if DynamicFlow[RecordNo] > FlowWasteAvg*12/NoWasteInts then
    if no_waste_ints > 0 and dynamic_flow[record_no] > flow_waste_avg * no_diurnal_ints / no_waste_ints:
        # Toggle the wastage flag
        wastage_on[record_no] = not wastage_on[record_no]
        changed = True

        # Update the count
        if wastage_on[record_no]:
            no_waste_ints += 1
        else:
            no_waste_ints -= 1

        # Redistribute wastage flow across active intervals
        for i in range(1, no_diurnal_ints + 1):
            if wastage_on[i]:
                flow_waste[i] = flow_waste_avg * no_diurnal_ints / no_waste_ints
            else:
                flow_waste[i] = 0.0

    return {
        "WastageOn": wastage_on,
        "FlowWaste": flow_waste,
        "NoWasteInts": no_waste_ints,
        "Changed": changed,
    }


# ---------------------------------------------------------------------------
# 4.  run_diurnal  --  ported from DIURNAL.PAS Procedure Diurnal
# ---------------------------------------------------------------------------

def run_diurnal(
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    ww_params: WastewaterParams,
    integration_params: IntegrationParams,
    C_steady: npt.NDArray[np.float64],
    diurnal_data: list[DiurnalData] | list[dict[str, float]],
    stoich_matrix: npt.NDArray[np.float64] | None = None,
    max_cycles: int = 50,
    convergence_threshold: float = 0.007,
) -> dict[str, Any]:
    """Run the diurnal (dynamic) simulation.

    This is a faithful port of ``Procedure Diurnal`` from DIURNAL.PAS.
    It cycles through 24-hour periods, each divided into 12 two-hourly
    influent intervals.  Within each interval the integrator advances
    the reactor state by ``Steps`` data-integration sub-intervals.
    After each cycle the algorithm checks whether Xbh and Sbp in
    reactor 1 have converged to within *convergence_threshold* (0.7%
    by default) of their start-of-cycle values.  If converged (and
    cycle > 1), iteration stops.

    Parameters
    ----------
    plant_config : PlantConfig
        Reactor topology and operating parameters.
    kinetic_params : KineticParams
        Temperature-adjusted kinetic rate constants.
    stoich_params : StoichiometricParams
        Stoichiometric/yield coefficients.
    ww_params : WastewaterParams
        Influent wastewater parameters.  ``Sti`` and ``Nti`` will be
        overwritten each interval from *diurnal_data*.
    integration_params : IntegrationParams
        Numerical integration control (Accuracy, Theta, DataIntHours).
    C_steady : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Steady-state concentration matrix (initial conditions).  This
        array is **not** modified; a working copy is made internally.
    diurnal_data : list of DiurnalData or list of dict
        12 records, each with Time, Flow, COD, TKN -- the two-hourly
        flow-weighted influent pattern.  If dicts, keys must be
        ``"Time"``, ``"Flow"``, ``"COD"``, ``"TKN"``.
    stoich_matrix : ndarray, optional
        Pre-built stoichiometric matrix.  If *None*, built internally
        from *stoich_params*.
    max_cycles : int
        Maximum number of 24-hour cycles before giving up (default 50).
    convergence_threshold : float
        Relative convergence tolerance for Xbh and Sbp between start
        and end of a cycle (default 0.007, i.e. 0.7%).

    Returns
    -------
    dict with keys:

    - ``"response"`` : list of ndarray or None
      1-based list.  ``response[k]`` is shape ``(NO_DI_VARS+1, data_per_day+1)``
      for reactor *k*.  Index 0 is None.  Contains the final cycle's
      time-series.
    - ``"cycle_count"`` : int
      Number of cycles executed.
    - ``"converged"`` : bool
      Whether the convergence criterion was met.
    - ``"C_final"`` : ndarray
      Concentration matrix at the end of the last cycle (before
      restoring steady-state values).
    - ``"data_per_day"`` : int
      Number of data points stored per 24-hour cycle.
    - ``"data_int_hours"`` : float
      Data integration interval in hours.
    """
    from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix

    # -- Normalise diurnal_data to list of DiurnalData --
    dd: list[DiurnalData] = []
    for rec in diurnal_data:
        if isinstance(rec, DiurnalData):
            dd.append(rec)
        elif isinstance(rec, dict):
            dd.append(DiurnalData(**rec))
        else:
            raise TypeError(
                f"diurnal_data records must be DiurnalData or dict, got {type(rec)}"
            )
    if len(dd) != NO_DIURNAL_INTS:
        raise ValueError(
            f"diurnal_data must have exactly {NO_DIURNAL_INTS} records, "
            f"got {len(dd)}"
        )

    # -- Build 1-based dynamic arrays (Pascal: DynamicFlow[1..12], etc.) --
    dynamic_flow = np.zeros(NO_DIURNAL_INTS + 1, dtype=np.float64)
    dynamic_sti = np.zeros(NO_DIURNAL_INTS + 1, dtype=np.float64)
    dynamic_nti = np.zeros(NO_DIURNAL_INTS + 1, dtype=np.float64)
    for idx, rec in enumerate(dd, start=1):
        dynamic_flow[idx] = rec.Flow
        dynamic_sti[idx] = rec.COD
        dynamic_nti[idx] = rec.TKN

    # -- Build stoichiometric matrix if not provided --
    if stoich_matrix is None:
        stoich_matrix = build_stoichiometric_matrix(stoich_params)

    # -- Set up wastage pattern (from SETWASTE.PAS via hydraulics.py) --
    waste_result = set_waste(
        no_diurnal_ints=NO_DIURNAL_INTS,
        dynamic_flow=dynamic_flow,
        flow_waste_avg=plant_config.FlowWasteAvg,
    )
    flow_waste_pattern: npt.NDArray[np.float64] = waste_result["FlowWaste"]

    # -- Timing constants (from DIURNAL.PAS) --
    data_int_hours = integration_params.DataIntHours
    inflow_int_hours = 24.0 / NO_DIURNAL_INTS  # 2.0 hours

    # Initialise integration parameters (step sizes, bounds)
    integration_parameters(integration_params, data_int_hours)

    # Steps per diurnal interval
    steps = round(inflow_int_hours / data_int_hours)
    # Data points per 24-hour cycle
    data_per_day = round(24.0 / data_int_hours)

    # -- Working concentration matrix (copy of steady state) --
    C = C_steady.copy()

    # -- Allocate response arrays: Response[k][var, data_no] --
    #    1-based list; k = 1..LastReactor, index 0 = None
    #    var = 1..18 (NO_DI_VARS), data_no = 0..data_per_day
    response: list[npt.NDArray[np.float64] | None] = [None]  # index 0 unused
    for _k in range(1, plant_config.LastReactor + 1):
        response.append(
            np.zeros((NO_DI_VARS + 1, data_per_day + 1), dtype=np.float64)
        )

    # -- Allocate workspace array for the integrator --
    shape_rc = (MAX_REAC_P1 + 1, TOTAL_COMPOUNDS + 1)
    MassRateIn = np.zeros(shape_rc, dtype=np.float64)

    # -- Save average inputs for restoration at end --
    sti_avg = ww_params.Sti
    nti_avg = ww_params.Nti
    flow_feed_avg = plant_config.FlowFeed
    flow_waste_avg_saved = plant_config.FlowWaste

    # -- Cycle loop --
    cycle_no = 0
    converged = False

    while True:
        cycle_no += 1
        data_no = 0

        # Compute air supply switching functions
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(
            plant_config, kinetic_params,
        )

        # Compute initial utilization rates and store response at data_no=0
        Oc, On, Ot, _ = utilization_rates(
            C, stoich_matrix, plant_config, kinetic_params, stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        store_response(
            C, Oc, On, Ot, stoich_params, ww_params,
            plant_config, response, 0,
        )

        # -- Iterate through 12 two-hourly intervals --
        for L in range(1, NO_DIURNAL_INTS + 1):
            # Set influent for this interval
            plant_config.FlowFeed = dynamic_flow[L]
            ww_params.Sti = dynamic_sti[L]
            ww_params.Nti = dynamic_nti[L]
            wastage_rate = flow_waste_pattern[L]

            # Fractionate influent for current Sti / Nti
            C0 = fractionate_influent(ww_params, stoich_params)

            # Compute dynamic flow division
            flow_div = flow_division_dynamic(
                last_reactor=plant_config.LastReactor,
                flow_feed=plant_config.FlowFeed,
                flow_ras_recycle=plant_config.FlowRASrecycle,
                flow_a_recycle=plant_config.FlowArecycle,
                flow_b_recycle=plant_config.FlowBrecycle,
                wastage_rate=wastage_rate,
                frac_feed=plant_config.FracFeed,
                vol=plant_config.Vol,
                flag_ras_in=plant_config.FlagRASIn,
                flag_a_in=plant_config.FlagAIn,
                flag_b_in=plant_config.FlagBIn,
                flag_a_out=plant_config.FlagAOut,
                flag_b_out=plant_config.FlagBOut,
            )

            DhOut = flow_div["DhOut"]
            DhFromPrevious = flow_div["DhFromPrevious"]
            DhFeed = flow_div["DhFeed"]
            DhRAS = flow_div["DhRAS"]
            DhArecycle = flow_div["DhArecycle"]
            DhBrecycle = flow_div["DhBrecycle"]

            # Compute air supply switching functions
            air_on_h, air_off_h, air_on_a, air_off_a = air_supply(
                plant_config, kinetic_params,
            )

            # -- Sub-intervals within this 2-hourly period --
            for m in range(1, steps + 1):
                integ_time = (L - 1) * inflow_int_hours + (m - 1) * data_int_hours
                integration_params.IntegTime = integ_time
                data_no += 1

                # Run integrator for one DataIntHours interval
                integrate(
                    C,
                    plant_config,
                    kinetic_params,
                    stoich_params,
                    integration_params,
                    stoich_matrix,
                    C0,
                    MassRateIn,
                    DhOut,
                    DhFeed,
                    DhRAS,
                    DhFromPrevious,
                    DhArecycle,
                    DhBrecycle,
                    air_on_h, air_off_h, air_on_a, air_off_a,
                    data_int_hours,
                )

                # Compute utilization rates after integration
                Oc, On, Ot, _ = utilization_rates(
                    C, stoich_matrix, plant_config, kinetic_params,
                    stoich_params,
                    air_on_h, air_off_h, air_on_a, air_off_a,
                )

                # Store time-series response
                store_response(
                    C, Oc, On, Ot, stoich_params, ww_params,
                    plant_config, response, data_no,
                )

        # -- Convergence check (from DIURNAL.PAS) --
        # Compare Xbh (compound 1) and Sbp (compound 5) at end vs start
        # of the cycle in reactor 1.
        #
        # Pascal:
        #   until (ABS(Response[1]^[1,DataPerDay]/Response[1]^[1,0]-1.0) < 0.007)
        #     AND (ABS(Response[1]^[5,DataPerDay]/Response[1]^[5,0]-1.0) < 0.007)
        #     AND (CycleNo > 1);
        xbh_start = response[1][1, 0]
        xbh_end = response[1][1, data_per_day]
        sbp_start = response[1][5, 0]
        sbp_end = response[1][5, data_per_day]

        xbh_converged = (
            xbh_start != 0.0
            and abs(xbh_end / xbh_start - 1.0) < convergence_threshold
        )
        sbp_converged = (
            sbp_start != 0.0
            and abs(sbp_end / sbp_start - 1.0) < convergence_threshold
        )

        if xbh_converged and sbp_converged and cycle_no > 1:
            converged = True
            break

        if cycle_no >= max_cycles:
            break

    # -- Restore average inputs (from SETAVG.PAS) --
    ww_params.Sti = sti_avg
    ww_params.Nti = nti_avg
    plant_config.FlowFeed = flow_feed_avg
    plant_config.FlowWaste = flow_waste_avg_saved
    # Re-fractionate with average values (side-effect only, result unused
    # here but mirrors the Pascal SetAvgInputs which calls FractionateInfluent)
    fractionate_influent(ww_params, stoich_params)

    # -- Preserve final dynamic concentrations before restoring SS --
    C_final = C.copy()

    # -- Restore steady-state concentrations (from DIURNAL.PAS) --
    # Pascal: for k := 1 to LastReactor do
    #           for i := 1 to (TotalCompounds-1) do C[k,i] := CSteady[k,i];
    for k in range(1, plant_config.LastReactor + 1):
        for i in range(1, TOTAL_COMPOUNDS):  # 1 .. 13
            C[k, i] = C_steady[k, i]

    return {
        "response": response,
        "cycle_count": cycle_no,
        "converged": converged,
        "C_final": C_final,
        "data_per_day": data_per_day,
        "data_int_hours": data_int_hours,
    }
