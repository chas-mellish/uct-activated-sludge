"""Two-timescale adaptive predictor-corrector ODE integrator.

Ported from the original Pascal source files:
  - INTEGRAT.PAS  -- Integrate procedure (predictor-corrector with adaptive
                     step sizing for slow and fast variables)
  - INTPARAM.PAS  -- CheckIntegParameters (step-size defaults)
  - DIURNAL.PAS   -- initial step-size setup

The algorithm uses nested loops:
  - Outer loop advances slow (particulate) variables with DeltaTLarge
  - Inner loop advances fast (soluble) variables with DeltaTSmall
  - Both loops use Euler prediction followed by trapezoidal correction,
    with local-error estimation and cube-root step-size adaptation.

All arrays use 1-based indexing (index 0 is unused padding) to match
the Pascal original and simplify cross-validation.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from uct_activated_sludge.constants import (
    FIRST_SLOW,
    LAST_SLOW,
    FIRST_RAPID,
    LAST_RAPID,
    LAST_COMPOUND,
    NO_PROCESSES,
    TOTAL_COMPOUNDS,
    MAX_REAC_P1,
    NO_PART,
)
from uct_activated_sludge.kinetics import process_rates
from uct_activated_sludge.models import (
    IntegrationParams,
    KineticParams,
    PlantConfig,
    StoichiometricParams,
)

__all__ = [
    "integrate",
    "integration_parameters",
]


# ---------------------------------------------------------------------------
# Helper: compute error-tolerance vector Epsilon
# ---------------------------------------------------------------------------

def _compute_epsilon(
    C: npt.NDArray[np.float64],
    accuracy: float,
) -> npt.NDArray[np.float64]:
    """Compute per-compound relative error tolerances.

    From DIURNAL.PAS:
        for i := FirstSlow to LastRapid do
          if C[1,i] <> 0.0
            then Epsilon[i] := C[1,i]*Accuracy/100
            else Epsilon[i] := 1.0e06;

    Parameters
    ----------
    C : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Concentration matrix (1-based).
    accuracy : float
        Accuracy parameter (e.g. 0.50).

    Returns
    -------
    Epsilon : ndarray, shape (TOTAL_COMPOUNDS+1,)
        1-based error-tolerance vector.
    """
    Epsilon = np.zeros(TOTAL_COMPOUNDS + 1, dtype=np.float64)
    for i in range(FIRST_SLOW, LAST_RAPID + 1):
        if C[1, i] != 0.0:
            Epsilon[i] = C[1, i] * accuracy / 100.0
        else:
            Epsilon[i] = 1.0e06
    # Floor at 1e-06 to avoid division issues
    for i in range(FIRST_SLOW, LAST_RAPID + 1):
        if abs(Epsilon[i]) < 1.0e-06:
            Epsilon[i] = 1.0e-06
    return Epsilon


# ---------------------------------------------------------------------------
# Sub-operations (nested procedures from INTEGRAT.PAS)
# ---------------------------------------------------------------------------

def _input_mass_rates(
    C: npt.NDArray[np.float64],
    C0: npt.NDArray[np.float64],
    last_reactor: int,
    first: int,
    last: int,
    DhFromPrevious: npt.NDArray[np.float64],
    DhRAS: npt.NDArray[np.float64],
    DhArecycle: npt.NDArray[np.float64],
    DhBrecycle: npt.NDArray[np.float64],
    DhFeed: npt.NDArray[np.float64],
    reactor_a_out: int,
    reactor_b_out: int,
    MassRateIn: npt.NDArray[np.float64],
) -> None:
    """Compute mass input rates for each reactor (Input procedure).

    From INTEGRAT.PAS:
        MassRateIn[K,I] := DhFromPrevious[K]*C[K-1,I]
                           + DhRAS[K,I]*C[LastReactor,I]
                           + DhArecycle[K]*C[ReactorAOut,I]
                           + DhBrecycle[K]*C[ReactorBOut,I]
                           + DhFeed[K]*C0[I]

    Modifies MassRateIn in-place.
    """
    for i in range(first, last + 1):
        for k in range(1, last_reactor + 1):
            MassRateIn[k, i] = (
                DhFromPrevious[k] * C[k - 1, i]
                + DhRAS[k, i] * C[last_reactor, i]
                + DhArecycle[k] * C[reactor_a_out, i]
                + DhBrecycle[k] * C[reactor_b_out, i]
                + DhFeed[k] * C0[i]
            )


def _react(
    C: npt.NDArray[np.float64],
    k: int,
    first: int,
    last: int,
    stoich_matrix: npt.NDArray[np.float64],
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
    Rate: npt.NDArray[np.float64],
) -> None:
    """Compute reaction rates for reactor k (React procedure).

    From INTEGRAT.PAS:
        ProcessRates;
        FOR I:=First TO Last DO Rate[K,I]:=0.0;
        FOR I:=First TO Last DO
          FOR J:=1 TO NoProcesses DO
            IF Stoich[I,J]<>0.0 THEN Rate[K,I]:=Rate[K,I]+Stoich[I,J]*Rho[J];

    Modifies Rate in-place.
    """
    Rho = process_rates(
        C, k, kinetic_params, stoich_params,
        air_on_hetero, air_off_hetero,
        air_on_auto, air_off_auto,
    )
    for i in range(first, last + 1):
        Rate[k, i] = 0.0
    for i in range(first, last + 1):
        for j in range(1, NO_PROCESSES + 1):
            if stoich_matrix[i, j] != 0.0:
                Rate[k, i] += stoich_matrix[i, j] * Rho[j]


def _derivative(
    C: npt.NDArray[np.float64],
    C0: npt.NDArray[np.float64],
    last_reactor: int,
    first: int,
    last: int,
    DhFromPrevious: npt.NDArray[np.float64],
    DhRAS: npt.NDArray[np.float64],
    DhArecycle: npt.NDArray[np.float64],
    DhBrecycle: npt.NDArray[np.float64],
    DhFeed: npt.NDArray[np.float64],
    DhOut: npt.NDArray[np.float64],
    reactor_a_out: int,
    reactor_b_out: int,
    stoich_matrix: npt.NDArray[np.float64],
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
    MassRateIn: npt.NDArray[np.float64],
    Rate: npt.NDArray[np.float64],
    dCdt: npt.NDArray[np.float64],
) -> None:
    """Compute derivatives for all reactors (Derivative procedure).

    From INTEGRAT.PAS:
        Input(First,Last);
        FOR K:=1 TO LastReactor DO
          BEGIN
            React(First,Last);
            FOR I:=First TO Last DO
              dCdt[K,I]:=MassRateIn[K,I]-DhOut[K]*C[K,I]+Rate[K,I];
          END;

    Modifies MassRateIn, Rate, and dCdt in-place.
    """
    _input_mass_rates(
        C, C0, last_reactor, first, last,
        DhFromPrevious, DhRAS, DhArecycle, DhBrecycle, DhFeed,
        reactor_a_out, reactor_b_out, MassRateIn,
    )
    for k in range(1, last_reactor + 1):
        _react(
            C, k, first, last, stoich_matrix,
            kinetic_params, stoich_params,
            air_on_hetero, air_off_hetero, air_on_auto, air_off_auto,
            Rate,
        )
        for i in range(first, last + 1):
            dCdt[k, i] = MassRateIn[k, i] - DhOut[k] * C[k, i] + Rate[k, i]


def _predict(
    C: npt.NDArray[np.float64],
    dCdt: npt.NDArray[np.float64],
    H: float,
    last_reactor: int,
    first: int,
    last: int,
    CStart: npt.NDArray[np.float64],
    dCdtStart: npt.NDArray[np.float64],
) -> None:
    """Euler forward prediction (Predict procedure).

    From INTEGRAT.PAS:
        FOR K:=1 TO LastReactor DO
          FOR I:=First TO Last DO
            CStart[K,I]:=C[K,I];
            dCdtStart[K,I]:=dCdt[K,I];
            C[K,I]:=C[K,I]+H*dCdt[K,I];

    Modifies C, CStart, dCdtStart in-place.
    """
    for k in range(1, last_reactor + 1):
        for i in range(first, last + 1):
            CStart[k, i] = C[k, i]
            dCdtStart[k, i] = dCdt[k, i]
            C[k, i] = C[k, i] + H * dCdt[k, i]


def _correct(
    C: npt.NDArray[np.float64],
    dCdt: npt.NDArray[np.float64],
    H: float,
    last_reactor: int,
    first: int,
    last: int,
    CStart: npt.NDArray[np.float64],
    dCdtStart: npt.NDArray[np.float64],
    CCheck: npt.NDArray[np.float64],
    LocalEr: npt.NDArray[np.float64],
) -> None:
    """Trapezoidal correction (Correct procedure).

    From INTEGRAT.PAS:
        FOR K:=1 TO LastReactor DO
          FOR I:=First TO Last DO
            CCheck[K,I]:=C[K,I];
            C[K,I]:=H/2*(dCdtStart[K,I]+dCdt[K,I])+CStart[K,I];
            LocalEr[K,I]:=1/5*ABS(CCheck[K,I]-C[K,I]);

    Modifies C, CCheck, LocalEr in-place.
    """
    for k in range(1, last_reactor + 1):
        for i in range(first, last + 1):
            CCheck[k, i] = C[k, i]
            C[k, i] = H / 2.0 * (dCdtStart[k, i] + dCdt[k, i]) + CStart[k, i]
            LocalEr[k, i] = (1.0 / 5.0) * abs(CCheck[k, i] - C[k, i])


def _calc_local_error(
    LocalEr: npt.NDArray[np.float64],
    Epsilon: npt.NDArray[np.float64],
    last_reactor: int,
    first: int,
    last: int,
) -> float:
    """Compute maximum local error across all reactors and compounds.

    From INTEGRAT.PAS:
        LocalError :=1.0e-12;
        FOR K:=1 TO LastReactor DO
          FOR I:=First TO Last DO
            IF (LocalEr[K,I]/Epsilon[I]) > LocalError THEN
              LocalError := LocalEr[K,I]/Epsilon[I];

    Returns
    -------
    float
        The maximum local error ratio.
    """
    local_error = 1.0e-12
    for k in range(1, last_reactor + 1):
        for i in range(first, last + 1):
            ratio = LocalEr[k, i] / Epsilon[i]
            if ratio > local_error:
                local_error = ratio
    return local_error


def _preserve_variable(
    C: npt.NDArray[np.float64],
    last_reactor: int,
    first: int,
    last: int,
    CPreserved: npt.NDArray[np.float64],
) -> None:
    """Copy C[k,first..last] into CPreserved for all reactors.

    From INTEGRAT.PAS PreserveVariable procedure.
    """
    for k in range(1, last_reactor + 1):
        for i in range(first, last + 1):
            CPreserved[k, i] = C[k, i]


def _recover_variable(
    C: npt.NDArray[np.float64],
    last_reactor: int,
    first: int,
    last: int,
    CPreserved: npt.NDArray[np.float64],
) -> None:
    """Copy CPreserved[k,first..last] back into C for all reactors.

    From INTEGRAT.PAS RecoverVariable procedure.
    """
    for k in range(1, last_reactor + 1):
        for i in range(first, last + 1):
            C[k, i] = CPreserved[k, i]


def _interpolate_slow(
    C: npt.NDArray[np.float64],
    CStartStep: npt.NDArray[np.float64],
    dCdtStart: npt.NDArray[np.float64],
    TSmall: float,
    last_reactor: int,
    first: int,
    last: int,
) -> None:
    """Linearly interpolate slow variables at fractional position TSmall.

    From INTEGRAT.PAS Interpolate procedure:
        FOR K:=1 TO LastReactor DO
          FOR I:=First TO Last DO
            C[K,I]:=CStartStep[K,I]+dCdtStart[K,I]*TSmall;

    Note: TSmall here is the absolute inner-loop time, not a fraction.
    The interpolation uses the derivative at the start of the large step
    to estimate slow variable values at the current inner-loop time.
    """
    for k in range(1, last_reactor + 1):
        for i in range(first, last + 1):
            C[k, i] = CStartStep[k, i] + dCdtStart[k, i] * TSmall


def _adapt_step_size(
    delta_t: float,
    local_error: float,
    theta: float,
    min_dt: float,
    max_dt: float,
) -> float:
    """Adapt step size using cube-root scaling.

    From INTEGRAT.PAS:
        DeltaT := DeltaT * EXP(1/3 * LN(Theta/LocalError))

    The step size is clamped to [min_dt, max_dt].
    """
    if local_error > 0.0:
        ratio = theta / local_error
        if ratio > 0.0:
            factor = math.exp((1.0 / 3.0) * math.log(ratio))
            delta_t *= factor
    # Clamp to bounds
    if max_dt > 0.0:
        delta_t = min(delta_t, max_dt)
    if min_dt > 0.0:
        delta_t = max(delta_t, min_dt)
    return delta_t


# ---------------------------------------------------------------------------
# integration_parameters  (from INTPARAM.PAS / DIURNAL.PAS)
# ---------------------------------------------------------------------------

def integration_parameters(
    integration_params: IntegrationParams,
    data_int_hours: float,
) -> IntegrationParams:
    """Set default step sizes based on data integration interval.

    Initialises DeltaTlarge, DeltaTsmall, IntegIntDays, and the
    min/max bounds from the data integration interval.

    From DIURNAL.PAS:
        IntegIntDays := DataIntHours/24.0;
        DeltaTlarge := DataIntHours/60/24;
        DeltaTsmall := DeltaTlarge/5;

    Parameters
    ----------
    integration_params : IntegrationParams
        Integration control parameters (modified in-place and returned).
    data_int_hours : float
        Data integration interval in hours.

    Returns
    -------
    IntegrationParams
        The same object, with step-size fields populated.
    """
    ip = integration_params
    ip.DataIntHours = data_int_hours

    # Data integration interval in days
    ip.IntegIntDays = data_int_hours / 24.0

    # Initial step sizes (from DIURNAL.PAS)
    ip.DeltaTlarge = data_int_hours / 60.0 / 24.0
    ip.DeltaTsmall = ip.DeltaTlarge / 5.0

    # Max step sizes: the large step cannot exceed the integration interval;
    # the small step cannot exceed the large step.
    ip.MaxDeltaTLarge = ip.IntegIntDays
    ip.MaxDeltaTSmall = ip.DeltaTlarge

    # Min step sizes: small fractions to prevent step collapse
    ip.MinDeltaTLarge = ip.DeltaTlarge / 1000.0
    ip.MinDeltaTSmall = ip.DeltaTsmall / 1000.0

    # Reset truncation state
    ip.TruncatedLarge = False
    ip.TruncatedSmall = False
    ip.PrevDeltaTLarge = 0.0
    ip.PrevDeltaTSmall = 0.0

    return ip


# ---------------------------------------------------------------------------
# Main integration routine (from INTEGRAT.PAS)
# ---------------------------------------------------------------------------

def integrate(
    C: npt.NDArray[np.float64],
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    integration_params: IntegrationParams,
    stoich_matrix: npt.NDArray[np.float64],
    C0: npt.NDArray[np.float64],
    MassRateIn: npt.NDArray[np.float64],
    DhOut: npt.NDArray[np.float64],
    DhFeed: npt.NDArray[np.float64],
    DhRAS: npt.NDArray[np.float64],
    DhFromPrevious: npt.NDArray[np.float64],
    DhArecycle: npt.NDArray[np.float64],
    DhBrecycle: npt.NDArray[np.float64],
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
    data_int_hours: float,
) -> npt.NDArray[np.float64]:
    """Advance the reactor state over one data integration interval.

    This is a faithful port of the Integrate procedure from INTEGRAT.PAS.
    It uses a two-timescale adaptive predictor-corrector algorithm:

    - Outer loop: advances slow (particulate) variables (indices 1-7)
      with step size DeltaTLarge.
    - Inner loop: advances fast (soluble) variables (indices 8-13)
      with step size DeltaTSmall, nested inside each large step.

    Both loops use Euler prediction followed by trapezoidal correction,
    with local-error estimation and cube-root step-size adaptation.

    Parameters
    ----------
    C : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Concentration matrix (1-based). Modified in-place.
    plant_config : PlantConfig
        Reactor topology parameters.
    kinetic_params : KineticParams
        Temperature-adjusted kinetic rate constants.
    stoich_params : StoichiometricParams
        Stoichiometric/yield coefficients.
    integration_params : IntegrationParams
        Integration control parameters (DeltaTlarge, DeltaTsmall, etc.).
        Modified in-place to carry state between calls.
    stoich_matrix : ndarray, shape (TotalCompounds+1, NoProcesses+1)
        Stoichiometric matrix.
    C0 : ndarray, shape (TotalCompounds+1,)
        Influent concentration vector (1-based).
    MassRateIn : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Mass input rate workspace (modified in-place).
    DhOut : ndarray, shape (MaxReacP1+1,)
        Dilution rate for outflow from each reactor (1-based).
    DhFeed : ndarray, shape (MaxReacP1+1,)
        Dilution rate for feed to each reactor (1-based).
    DhRAS : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        RAS dilution rate per reactor and compound (1-based).
    DhFromPrevious : ndarray, shape (MaxReacP1+1,)
        Dilution rate for flow from previous reactor (1-based).
    DhArecycle : ndarray, shape (MaxReacP1+1,)
        Dilution rate for a-recycle to each reactor (1-based).
    DhBrecycle : ndarray, shape (MaxReacP1+1,)
        Dilution rate for b-recycle to each reactor (1-based).
    air_on_hetero, air_off_hetero : ndarray
        Heterotrophic DO switching functions per reactor (1-based).
    air_on_auto, air_off_auto : ndarray
        Autotrophic DO switching functions per reactor (1-based).
    data_int_hours : float
        Data integration interval in hours.

    Returns
    -------
    ndarray
        The updated C matrix (same object as input, modified in-place).
    """
    ip = integration_params
    last_reactor = plant_config.LastReactor
    reactor_a_out = plant_config.ReactorAOut
    reactor_b_out = plant_config.ReactorBOut

    # Integration interval in days
    IntegIntDays = ip.IntegIntDays

    # Shorthand references
    DeltaTLarge = ip.DeltaTlarge
    DeltaTSmall = ip.DeltaTsmall
    Theta = ip.Theta

    # Compute error tolerances (from DIURNAL.PAS)
    Epsilon = _compute_epsilon(C, ip.Accuracy)

    # Allocate workspace arrays (1-based, matching Pascal types)
    shape_rc = (MAX_REAC_P1 + 1, TOTAL_COMPOUNDS + 1)
    dCdt = np.zeros(shape_rc, dtype=np.float64)
    Rate = np.zeros(shape_rc, dtype=np.float64)
    CStart = np.zeros(shape_rc, dtype=np.float64)
    dCdtStart = np.zeros(shape_rc, dtype=np.float64)
    CCheck = np.zeros(shape_rc, dtype=np.float64)
    LocalEr = np.zeros(shape_rc, dtype=np.float64)
    CStartStep = np.zeros(shape_rc, dtype=np.float64)
    CEnd = np.zeros(shape_rc, dtype=np.float64)

    # Common derivative arguments (to avoid repetition)
    def compute_deriv(first: int, last: int) -> None:
        _derivative(
            C, C0, last_reactor, first, last,
            DhFromPrevious, DhRAS, DhArecycle, DhBrecycle, DhFeed, DhOut,
            reactor_a_out, reactor_b_out, stoich_matrix,
            kinetic_params, stoich_params,
            air_on_hetero, air_off_hetero, air_on_auto, air_off_auto,
            MassRateIn, Rate, dCdt,
        )

    # -----------------------------------------------------------------------
    # Time tracking
    # -----------------------------------------------------------------------
    TLarge = 0.0
    TSmall = 0.0

    # Restore DeltaTLarge from previous truncation if needed
    TruncatedLarge = ip.TruncatedLarge
    TruncatedSmall = ip.TruncatedSmall
    PrevDeltaTLarge = ip.PrevDeltaTLarge
    PrevDeltaTSmall = ip.PrevDeltaTSmall

    if TruncatedLarge:
        DeltaTLarge = PrevDeltaTLarge

    # Track local error for CheckErrorSol reset of DeltaTSmall
    LocalError = 0.0

    # ===================================================================
    # OUTER LOOP: slow variables (repeat ... until TLarge >= 0.999*IntegIntDays)
    # ===================================================================
    while True:
        TruncatedLarge = False

        # Truncate DeltaTLarge to hit the end of the integration interval
        if (TLarge + DeltaTLarge) > IntegIntDays:
            TruncatedLarge = True
            PrevDeltaTLarge = DeltaTLarge
            DeltaTLarge = IntegIntDays - TLarge

        TLarge += DeltaTLarge

        # Compute derivatives for slow variables
        compute_deriv(FIRST_SLOW, LAST_SLOW)

        # Preserve all variables (slow + rapid) at start of large step
        _preserve_variable(C, last_reactor, FIRST_SLOW, LAST_RAPID, CStartStep)

        # Predict slow variables (Euler forward)
        _predict(C, dCdt, DeltaTLarge, last_reactor, FIRST_SLOW, LAST_SLOW,
                 CStart, dCdtStart)

        # Save predicted slow values (CEnd) before interpolation overwrites them
        _preserve_variable(C, last_reactor, FIRST_SLOW, LAST_SLOW, CEnd)
        # Recover slow variables to start-of-step values for interpolation
        _recover_variable(C, last_reactor, FIRST_SLOW, LAST_SLOW, CStartStep)

        # Restore DeltaTSmall from previous truncation if needed
        if TruncatedSmall:
            DeltaTSmall = PrevDeltaTSmall
        # Reset DeltaTSmall if previous outer step was rejected
        if LocalError >= 1.0:
            DeltaTSmall = 0.64 / 60.0 / 24.0

        # ---------------------------------------------------------------
        # INNER LOOP: fast variables (repeat ... until TSmall >= 0.999*TLarge)
        # ---------------------------------------------------------------
        while True:
            TruncatedSmall = False

            # Truncate DeltaTSmall to hit the end of the large step
            if (TSmall + DeltaTSmall) > TLarge:
                TruncatedSmall = True
                PrevDeltaTSmall = DeltaTSmall
                DeltaTSmall = TLarge - TSmall

            TSmall += DeltaTSmall

            # Compute derivatives for fast variables
            compute_deriv(FIRST_RAPID, LAST_RAPID)

            # Predict fast variables (Euler forward)
            _predict(C, dCdt, DeltaTSmall, last_reactor,
                     FIRST_RAPID, LAST_RAPID, CStart, dCdtStart)

            # Interpolate slow variables at current TSmall
            _interpolate_slow(C, CStartStep, dCdtStart, TSmall,
                              last_reactor, FIRST_SLOW, LAST_SLOW)

            # Recompute derivatives for fast variables with interpolated slow vars
            compute_deriv(FIRST_RAPID, LAST_RAPID)

            # Correct fast variables (trapezoidal)
            _correct(C, dCdt, DeltaTSmall, last_reactor,
                     FIRST_RAPID, LAST_RAPID, CStart, dCdtStart,
                     CCheck, LocalEr)

            # Compute local error for fast variables
            LocalError = _calc_local_error(
                LocalEr, Epsilon, last_reactor, FIRST_RAPID, LAST_RAPID,
            )

            # CheckErrorSol: adapt DeltaTSmall
            if LocalError >= 1.0:
                # Step rejected: rewind TSmall, adapt step, recover state
                TSmall -= DeltaTSmall
                DeltaTSmall = _adapt_step_size(
                    DeltaTSmall, LocalError, Theta,
                    ip.MinDeltaTSmall, ip.MaxDeltaTSmall,
                )
                _recover_variable(C, last_reactor, FIRST_RAPID, LAST_RAPID,
                                  CStart)
            else:
                # Step accepted: adapt step for next iteration
                DeltaTSmall = _adapt_step_size(
                    DeltaTSmall, LocalError, Theta,
                    ip.MinDeltaTSmall, ip.MaxDeltaTSmall,
                )

            # Check inner loop termination
            if TSmall >= 0.999 * TLarge:
                break

        # ---------------------------------------------------------------
        # After inner loop: correct slow variables
        # ---------------------------------------------------------------
        # Recover predicted slow values from CEnd
        _recover_variable(C, last_reactor, FIRST_SLOW, LAST_SLOW, CEnd)

        # Recompute derivatives for slow variables
        compute_deriv(FIRST_SLOW, LAST_SLOW)

        # Correct slow variables (trapezoidal)
        _correct(C, dCdt, DeltaTLarge, last_reactor,
                 FIRST_SLOW, LAST_SLOW, CStart, dCdtStart,
                 CCheck, LocalEr)

        # Compute local error for slow variables
        LocalError = _calc_local_error(
            LocalEr, Epsilon, last_reactor, FIRST_SLOW, LAST_SLOW,
        )

        # CheckError: adapt DeltaTLarge
        if LocalError >= 1.0:
            # Step rejected: rewind time, adapt step, recover state
            TLarge -= DeltaTLarge
            TSmall -= DeltaTLarge
            DeltaTLarge = _adapt_step_size(
                DeltaTLarge, LocalError, Theta,
                ip.MinDeltaTLarge, ip.MaxDeltaTLarge,
            )
            _recover_variable(C, last_reactor, FIRST_SLOW, LAST_RAPID,
                              CStartStep)
        else:
            # Step accepted: adapt step for next iteration
            DeltaTLarge = _adapt_step_size(
                DeltaTLarge, LocalError, Theta,
                ip.MinDeltaTLarge, ip.MaxDeltaTLarge,
            )

        # Check outer loop termination
        if TLarge >= 0.999 * IntegIntDays:
            break

    # -----------------------------------------------------------------------
    # Persist adaptive state for next call
    # -----------------------------------------------------------------------
    ip.DeltaTlarge = DeltaTLarge
    ip.DeltaTsmall = DeltaTSmall
    ip.TruncatedLarge = TruncatedLarge
    ip.TruncatedSmall = TruncatedSmall
    ip.PrevDeltaTLarge = PrevDeltaTLarge
    ip.PrevDeltaTSmall = PrevDeltaTSmall

    return C
