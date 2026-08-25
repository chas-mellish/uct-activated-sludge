"""Steady-state simulation driver for the UCT Activated Sludge Model.

Ported from the original Pascal source files:
  - STDYUNIT.PAS   -- SteadyState, SteadySolution, SeedValues, StoreAvgConcs
  - STDYRSLT.PAS   -- OutputSteadyState (formatted text output)
  - WASTE.PAS      -- WastageAndFlows (iterative tracer-based wastage calc)
  - FLOWSS.PAS     -- FlowDivisionSS  (steady-state flow division)
  - SCALE.PAS      -- ScaleValues
  - NEWTON.PAS     -- Newton solver

The main entry point is :func:`run_steady_state`, which orchestrates the
full steady-state simulation: influent fractionation, temperature
adjustment, wastage calculation, seed-value generation, Newton solve,
and utilization-rate computation.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from uct_activated_sludge.constants import (
    ASM1Component as Comp,
    COMPOUND_NAMES,
    COMPOUND_UNITS,
    LAST_COMPOUND,
    MAX_REAC_P1,
    NO_PART,
    TOTAL_COMPOUNDS,
)
from uct_activated_sludge.hydraulics import (
    flow_division_ss_inplace,
    wastage_and_flows,
)
from uct_activated_sludge.influent import fractionate_influent
from uct_activated_sludge.kinetics import air_supply, utilization_rates
from uct_activated_sludge.models import (
    IntegrationParams,
    KineticParams,
    PlantConfig,
    StoichiometricParams,
    WastewaterParams,
    _zeros_1based,
    _zeros_2d_1based,
)
from uct_activated_sludge.solver import newton, scale_values
from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix
from uct_activated_sludge.temperature import adjust_temperature

__all__ = [
    "run_steady_state",
    "seed_values",
    "output_steady_state",
    "store_avg_concs",
]


# ---------------------------------------------------------------------------
# 1. seed_values  --  ported from SeedValues in STDYUNIT.PAS
# ---------------------------------------------------------------------------

def seed_values(
    C: npt.NDArray[np.float64],
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    ww_params: WastewaterParams,
    C0: npt.NDArray[np.float64],
    tracer: npt.NDArray[np.float64],
    mass_tracer: float,
) -> None:
    """Compute initial-guess (seed) concentrations for the Newton solver.

    Distributes biomass and substrate estimates across reactors based on
    tracer proportions from the wastage iteration.  This is a direct port
    of the ``SeedValues`` nested procedure in STDYUNIT.PAS.

    Parameters
    ----------
    C : ndarray, shape (MAX_REAC_P1+1, TOTAL_COMPOUNDS+1)
        Concentration matrix -- modified **in place**.
    plant_config : PlantConfig
        Reactor topology and operating parameters.
    kinetic_params : KineticParams
        Temperature-adjusted kinetic constants.
    stoich_params : StoichiometricParams
        Stoichiometric/yield parameters.
    ww_params : WastewaterParams
        Wastewater characterisation (Sti, Nti, fractions).
    C0 : ndarray
        Influent concentration vector (1-based).
    tracer : ndarray
        Tracer concentrations per reactor (1-based), from wastage calc.
    mass_tracer : float
        Total tracer mass (sum of tracer[k] * Vol[k] over reactors).
    """
    pc = plant_config
    sp = stoich_params

    Sti = ww_params.Sti
    Nti = ww_params.Nti
    Temp = pc.Temp
    Rs = pc.Rs
    fus = ww_params.Fus
    fup = ww_params.Fup
    Ixb = sp.Ixb
    Ya = sp.Ya
    FlowFeed = pc.FlowFeed

    # Temperature-adjusted decay rates for seed estimation
    # Pascal: bh24 := EXP(LN(0.24) + (Temp-20)*LN(1.029))
    bh24 = math.exp(math.log(0.24) + (Temp - 20) * math.log(1.029))
    bn04 = math.exp(math.log(0.04) + (Temp - 20) * math.log(1.029))

    # Total masses of biomass fractions
    MassXa = 0.66 * FlowFeed * Sti * (1 - fus - fup) * Rs / (1 + bh24 * Rs)
    MassXe = 0.2 * bh24 * Rs * MassXa
    MassXi = fup * FlowFeed * Sti * Rs

    # Nitrogen available for nitrifiers
    Ns = Ixb * (MassXa + MassXe + MassXi) / (Rs * FlowFeed)
    MassXn = FlowFeed * (Nti - Ns - 3.0) * Ya * Rs / (1 + bn04 * Rs)

    last_reactor = pc.LastReactor
    VolumeTotal = pc.VolumeTotal

    for k in range(1, last_reactor + 1):
        frac = tracer[k] * pc.Vol[k] / mass_tracer

        # Xbh (heterotrophic biomass)
        C[k, Comp.Xbh] = frac * MassXa / pc.Vol[k]
        # Xe (endogenous residue)
        C[k, Comp.Xe] = frac * MassXe / pc.Vol[k]
        # Xba (autotrophic biomass)
        C[k, Comp.Xba] = frac * MassXn / pc.Vol[k]

        # Xs (stored COD) -- depends on aeration and feed status
        if pc.ReactorAerated[k]:
            C[k, Comp.Xs] = 0.10 * C[k, Comp.Xbh]
        elif pc.FracFeed[k] > 0.0:
            C[k, Comp.Xs] = 0.50 * C[k, Comp.Xbh]
        else:
            C[k, Comp.Xs] = 0.20 * C[k, Comp.Xbh]

        # Sbp (particulate biodegradable COD)
        C[k, Comp.Sbp] = C[k, Comp.Xs]

        # Xi (particulate unbiodegradable COD)
        C[k, Comp.Xi] = frac * MassXi / pc.Vol[k]

        # Xnd (particulate biodegradable N)
        C[k, Comp.Xnd] = 10.0

        # Ss (soluble biodegradable COD)
        if (not pc.ReactorAerated[k]) and (pc.FracFeed[k] > 0.0):
            C[k, Comp.Ss] = 0.25 * C0[Comp.Ss]
        else:
            C[k, Comp.Ss] = 1.0

        # Snh (ammonia N)
        if pc.ReactorAerated[k]:
            if pc.FracFeed[k] > 0:
                C[k, Comp.Snh] = 0.5 * VolumeTotal / pc.Vol[k]
            else:
                C[k, Comp.Snh] = 0.5
        else:
            C[k, Comp.Snh] = 0.25 * (Nti - Ns)

        # Snd (soluble organic N)
        C[k, Comp.Snd] = 3.0

        # Sno (nitrate N)
        if pc.ReactorAerated[k]:
            C[k, Comp.Sno] = 1.0 * (Nti - Ns)
        else:
            C[k, Comp.Sno] = 0.001

        # Salk (alkalinity)
        C[k, Comp.Alk] = 0.5 * C0[Comp.Alk]

        # Si (soluble unbiodegradable COD)
        C[k, Comp.Si] = fus * Sti

    # Settling tank (pseudo-reactor at LastReactor+1):
    # Particulate compounds thickened by RAS ratio
    for i in range(1, NO_PART + 1):
        C[last_reactor + 1, i] = (
            (FlowFeed + pc.FlowRASrecycle - pc.FlowWaste)
            / pc.FlowRASrecycle
            * C[last_reactor, i]
        )
    # Soluble compounds pass through unchanged
    for i in range(NO_PART + 1, LAST_COMPOUND + 1):
        C[last_reactor + 1, i] = C[last_reactor, i]


# ---------------------------------------------------------------------------
# 2. output_steady_state  --  ported from OutputSteadyState in STDYRSLT.PAS
# ---------------------------------------------------------------------------

def output_steady_state(
    C: npt.NDArray[np.float64],
    C0: npt.NDArray[np.float64],
    plant_config: PlantConfig,
    stoich_params: StoichiometricParams,
    ww_params: WastewaterParams,
    Oc: npt.NDArray[np.float64],
    On: npt.NDArray[np.float64],
    Ot: npt.NDArray[np.float64],
    Denit: npt.NDArray[np.float64],
    inversions: int = 0,
) -> str:
    """Format steady-state results as a text report.

    Ported from ``OutputSteadyState`` in STDYRSLT.PAS.

    Parameters
    ----------
    C : ndarray
        Converged concentration matrix.
    C0 : ndarray
        Influent concentration vector (1-based).
    plant_config : PlantConfig
        Reactor topology.
    stoich_params : StoichiometricParams
        Stoichiometric parameters (CODVSS, VSSTSS used for VSS/TSS).
    ww_params : WastewaterParams
        Wastewater parameters (Fnu, Nti for TKN calculation).
    Oc, On, Ot, Denit : ndarray
        Per-reactor OUR and denitrification rates (1-based).
    inversions : int
        Number of Newton iterations to convergence.

    Returns
    -------
    str
        Multi-line formatted text report.
    """
    last_reactor = plant_config.LastReactor
    CODVSS = stoich_params.CODVSS
    VSSTSS = ww_params.VSSTSS
    Fnu = ww_params.Fnu
    Nti = ww_params.Nti

    lines: list[str] = []

    lines.append(
        f"  *****  STEADY STATE RESULTS *****"
        f"              Inversions = {inversions}"
    )
    lines.append("")

    # Header
    header = "COMPOUND              INPUT           REACTOR"
    lines.append(header)
    reactor_nums = "                             "
    for k in range(1, last_reactor + 1):
        reactor_nums += f"{k:2d}     "
    lines.append(reactor_nums)

    # Compound rows (1 .. TotalCompounds-1, i.e. 1..13)
    for j in range(1, TOTAL_COMPOUNDS):  # 1..13
        name = COMPOUND_NAMES[j] if j < len(COMPOUND_NAMES) else f"Compound {j}"
        units = COMPOUND_UNITS[j] if j < len(COMPOUND_UNITS) else ""
        row = f"{name} =  {C0[j]:6.1f}"
        for k in range(1, last_reactor + 1):
            row += f"{C[k, j]:7.1f}"
        row += units
        lines.append(row)

    # Volatile Suspended Solids (VSS) and Total Suspended Solids (TSS)
    XvTot = _zeros_1based(MAX_REAC_P1)
    for k in range(1, last_reactor + 1):
        for i in range(1, 7):  # compounds 1..6
            XvTot[k] += C[k, i] / CODVSS

    lines.append("")
    row = "Volatile SS       =        "
    for k in range(1, last_reactor + 1):
        row += f"{XvTot[k]:7.1f}"
    row += " g VSS m-3 "
    lines.append(row)

    row = "Total SS          =        "
    for k in range(1, last_reactor + 1):
        if VSSTSS > 0:
            row += f"{XvTot[k] / VSSTSS:7.1f}"
        else:
            row += f"{'N/A':>7s}"
    row += " g TSS m-3 "
    lines.append(row)

    row = "OUR heterotrophs  =        "
    for k in range(1, last_reactor + 1):
        row += f"{Oc[k]:7.1f}"
    row += " g O2 m-3 h-1 "
    lines.append(row)

    row = "OUR autotrophs    =        "
    for k in range(1, last_reactor + 1):
        row += f"{On[k]:7.1f}"
    row += " g O2 m-3 h-1 "
    lines.append(row)

    row = "OUR total         =        "
    for k in range(1, last_reactor + 1):
        row += f"{Ot[k]:7.1f}"
    row += " g O2 m-3 h-1 "
    lines.append(row)

    row = "Denit. rate       =        "
    for k in range(1, last_reactor + 1):
        row += f"{Denit[k]:7.1f}"
    row += " g NO3-N m-3 h-1 "
    lines.append(row)

    # TKN = Snh + Snd + Fnu*Nti
    row = "TKN               =        "
    for k in range(1, last_reactor + 1):
        tkn = C[k, Comp.Snh] + C[k, Comp.Snd] + Fnu * Nti
        row += f"{tkn:7.1f}"
    row += " g N m-3"
    lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. store_avg_concs  --  ported from StoreAvgConcs in STDYUNIT.PAS
# ---------------------------------------------------------------------------

def store_avg_concs(
    C: npt.NDArray[np.float64],
    last_reactor: int,
) -> npt.NDArray[np.float64]:
    """Copy steady-state concentrations for use as diurnal initial conditions.

    Ported from ``StoreAvgConcs`` in STDYUNIT.PAS:

    .. code-block:: pascal

        For k := 1 to LastReactor do
          For i := 1 to (TotalCompounds - 1) do CSteady[k,i] := C[k,i];

    Parameters
    ----------
    C : ndarray
        Converged steady-state concentration matrix.
    last_reactor : int
        Number of active reactors.

    Returns
    -------
    ndarray
        Copy of C for indices [1..last_reactor, 1..TOTAL_COMPOUNDS-1].
        Same shape as C (full 2-D array with 1-based indexing).
    """
    CSteady = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
    for k in range(1, last_reactor + 1):
        for i in range(1, TOTAL_COMPOUNDS):  # 1 .. TotalCompounds-1 = 13
            CSteady[k, i] = C[k, i]
    return CSteady


# ---------------------------------------------------------------------------
# 4. run_steady_state  --  main orchestration
# ---------------------------------------------------------------------------

def run_steady_state(
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    ww_params: WastewaterParams,
    integration_params: IntegrationParams | None = None,
) -> dict:
    """Run the full steady-state simulation.

    Orchestrates the complete Pascal workflow from ``SteadyState`` and
    ``SteadySolution`` in STDYUNIT.PAS:

    1. Build stoichiometric matrix
    2. Adjust kinetic parameters for temperature
    3. Fractionate influent to get C0 vector
    4. Compute air-supply switching functions
    5. Initialise concentration matrix C
    6. Calculate wastage rate and flows (tracer Newton solve)
    7. Compute seed values (initial guesses for Newton)
    8. Compute steady-state flow division
    9. Scale values
    10. Run Newton solver (13 compounds)
    11. Compute utilization rates (OUR, denitrification)
    12. Format and return results

    Parameters
    ----------
    plant_config : PlantConfig
        Reactor topology and operating parameters.
    kinetic_params : KineticParams
        Kinetic rate constants (20 degC reference values).
    stoich_params : StoichiometricParams
        Stoichiometric/yield coefficients.
    ww_params : WastewaterParams
        Influent wastewater characterisation.
    integration_params : IntegrationParams, optional
        Not used directly by the steady-state solver, but accepted for
        API symmetry with the diurnal driver.

    Returns
    -------
    dict with keys:

    - ``"C"`` : ndarray -- converged concentration matrix
    - ``"C0"`` : ndarray -- influent concentration vector
    - ``"CSteady"`` : ndarray -- copy for diurnal initial conditions
    - ``"Stoich"`` : ndarray -- stoichiometric matrix
    - ``"Oc"`` : ndarray -- carbonaceous OUR per reactor
    - ``"On"`` : ndarray -- nitrogenous OUR per reactor
    - ``"Ot"`` : ndarray -- total OUR per reactor
    - ``"Denit"`` : ndarray -- denitrification rate per reactor
    - ``"air_on_hetero"`` : ndarray -- DO switching function
    - ``"air_off_hetero"`` : ndarray -- DO switching function
    - ``"air_on_auto"`` : ndarray -- DO switching function
    - ``"air_off_auto"`` : ndarray -- DO switching function
    - ``"kinetic_params"`` : KineticParams -- temperature-adjusted params
    - ``"plant_config"`` : PlantConfig -- updated plant config with flows
    - ``"converged"`` : bool -- whether Newton converged
    - ``"output"`` : str -- formatted text report
    - ``"FlowWaste"`` : float -- converged wastage flow rate
    """
    # Validate essential plant config parameters
    if plant_config.Rs <= 0.0:
        raise ValueError(
            "PlantConfig.Rs (sludge retention time) must be > 0. "
            f"Got {plant_config.Rs}."
        )
    if plant_config.FlowFeed <= 0.0:
        raise ValueError(
            "PlantConfig.FlowFeed must be > 0. "
            f"Got {plant_config.FlowFeed}."
        )
    if plant_config.VolumeTotal <= 0.0:
        raise ValueError(
            "PlantConfig.VolumeTotal must be > 0. "
            f"Got {plant_config.VolumeTotal}."
        )
    if plant_config.FlowRASrecycle <= 0.0:
        raise ValueError(
            "PlantConfig.FlowRASrecycle must be > 0. "
            f"Got {plant_config.FlowRASrecycle}."
        )

    # ---- Step 1: Build stoichiometric matrix ----
    Stoich = build_stoichiometric_matrix(stoich_params)

    # ---- Step 2: Temperature adjustment ----
    kp = adjust_temperature(kinetic_params, plant_config.Temp)

    # ---- Step 3: Fractionate influent ----
    C0 = fractionate_influent(ww_params, stoich_params)

    # ---- Step 4: Air supply switching functions ----
    air_on_hetero, air_off_hetero, air_on_auto, air_off_auto = air_supply(
        plant_config, kp
    )

    # ---- Step 5: Initialise concentration matrix ----
    C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)

    # ---- Step 6: Wastage and flows ----
    # The wastage calculation needs Newton and ScaleValues as callables
    # that operate on the shared C matrix.  We create closures that match
    # the signature expected by wastage_and_flows: fn(compounds).
    def _newton_fn(compounds: int) -> None:
        newton(
            C, compounds, plant_config, kp, stoich_params,
            Stoich, C0,
            air_on_hetero, air_off_hetero,
            air_on_auto, air_off_auto,
        )

    def _scale_fn(compounds: int) -> None:
        scale_values(C, plant_config, compounds)

    flow_waste = wastage_and_flows(
        plant=plant_config,
        C=C,
        newton_fn=_newton_fn,
        scale_values_fn=_scale_fn,
    )

    # Recover tracer concentrations and mass for seed values
    last_reactor = plant_config.LastReactor
    tracer = _zeros_1based(MAX_REAC_P1)
    for k in range(1, last_reactor + 2):
        tracer[k] = C[k, 1]
    mass_tracer = 0.0
    for k in range(1, last_reactor + 1):
        mass_tracer += tracer[k] * plant_config.Vol[k]

    # ---- Step 7: Seed values (initial guesses for full solve) ----
    seed_values(
        C, plant_config, kp, stoich_params, ww_params,
        C0, tracer, mass_tracer,
    )

    # ---- Step 8: Compute air supply (may have been affected by config) ----
    air_on_hetero, air_off_hetero, air_on_auto, air_off_auto = air_supply(
        plant_config, kp
    )

    # ---- Step 9: Steady-state flow division ----
    flow_division_ss_inplace(plant_config)

    # ---- Step 9b: Scale values ----
    _scale_fn(LAST_COMPOUND)

    # ---- Step 10: Newton solver (13 compounds) ----
    C, converged = newton(
        C, LAST_COMPOUND, plant_config, kp, stoich_params,
        Stoich, C0,
        air_on_hetero, air_off_hetero,
        air_on_auto, air_off_auto,
    )

    # ---- Step 11: Utilization rates ----
    Oc, On, Ot, Denit = utilization_rates(
        C, Stoich, plant_config, kp, stoich_params,
        air_on_hetero, air_off_hetero,
        air_on_auto, air_off_auto,
    )

    # ---- Step 12: Store steady-state concentrations ----
    CSteady = store_avg_concs(C, last_reactor)

    # ---- Step 13: Format output ----
    output_text = output_steady_state(
        C, C0, plant_config, stoich_params, ww_params,
        Oc, On, Ot, Denit,
    )

    return {
        "C": C,
        "C0": C0,
        "CSteady": CSteady,
        "Stoich": Stoich,
        "Oc": Oc,
        "On": On,
        "Ot": Ot,
        "Denit": Denit,
        "air_on_hetero": air_on_hetero,
        "air_off_hetero": air_off_hetero,
        "air_on_auto": air_on_auto,
        "air_off_auto": air_off_auto,
        "kinetic_params": kp,
        "plant_config": plant_config,
        "converged": converged,
        "output": output_text,
        "FlowWaste": flow_waste,
    }
