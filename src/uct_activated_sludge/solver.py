"""Newton-Raphson solver with Gauss elimination for the UCT Activated Sludge Model.

Ported from the original Pascal source files:
  - NEWTON.PAS   -- Newton procedure (SetUpEquations, CalculateJMat,
                     CalcNewX, Gauss, and the main Newton loop)
  - MACHEPS.PAS  -- CalcMachEps (replaced with numpy.finfo)
  - SCALE.PAS    -- ScaleValues procedure

All arrays use 1-based indexing (index 0 is unused padding) to match
the Pascal original and simplify cross-validation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from uct_activated_sludge.constants import (
    LAST_COMPOUND,
    MAX_N,
    MAX_REAC_P1,
    NO_PART,
    NO_PROCESSES,
    TOTAL_COMPOUNDS,
)
from uct_activated_sludge.kinetics import process_rates
from uct_activated_sludge.models import (
    KineticParams,
    PlantConfig,
    StoichiometricParams,
)

__all__ = [
    "newton",
    "gauss",
    "scale_values",
]

# Machine epsilon for float64 (replaces Pascal CalcMachEps)
MACHEPS: float = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------
# ScaleValues  (from SCALE.PAS)
# ---------------------------------------------------------------------------

def scale_values(
    C: npt.NDArray[np.float64],
    plant_config: PlantConfig,
    compounds: int = LAST_COMPOUND,
) -> npt.NDArray[np.float64]:
    """Compute scaling vector from current concentrations.

    Translates the Pascal ``ScaleValues`` procedure from SCALE.PAS.

    Parameters
    ----------
    C : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Concentration matrix.  ``C[k, i]`` is compound *i* in reactor *k*.
    plant_config : PlantConfig
        Reactor topology (``LastReactor``).
    compounds : int
        Number of compounds being solved (1 for tracer, 13 for full).

    Returns
    -------
    Scale : ndarray, shape (MaxN+1,)
        1-based scaling vector.  ``Scale[L]`` is the inverse of the
        corresponding concentration, or 1.0 when the concentration is zero.
    """
    last_reactor = plant_config.LastReactor
    scale = np.zeros(MAX_N + 1, dtype=np.float64)

    for k in range(1, last_reactor + 2):  # 1 .. LastReactor+1
        for i in range(1, compounds + 1):
            idx = compounds * (k - 1) + i
            if C[k, i] == 0.0:
                scale[idx] = 1.0
            else:
                scale[idx] = 1.0 / C[k, i]

    return scale


# ---------------------------------------------------------------------------
# Gauss elimination with partial pivoting  (from NEWTON.PAS)
# ---------------------------------------------------------------------------

def gauss(
    n: int,
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Solve the linear system A x = b via Gaussian elimination with
    partial pivoting (permuted Gauss method).

    Directly translated from the Pascal ``Gauss`` procedure in NEWTON.PAS.

    Parameters
    ----------
    n : int
        System size (number of equations / unknowns).
    A : ndarray, shape (n+1, n+1) or larger
        Coefficient matrix (1-based indexing, row/col 0 unused).
        **Modified in place** during elimination.
    b : ndarray, shape (n+1,) or larger
        Right-hand-side vector (1-based).  **Modified in place**.

    Returns
    -------
    x : ndarray, shape (n+1,) or larger
        Solution vector (1-based).
    success : bool
        True if the system was solved; False if a near-zero pivot was
        encountered (singular or near-singular matrix).

    Notes
    -----
    The ``assumedzero`` threshold is 1E-20, matching the Pascal original.
    """
    ASSUMED_ZERO = 1.0e-20

    x = np.zeros_like(b)
    # Permutation vector (1-based)
    p = np.arange(n + 1, dtype=np.int64)  # p[0] unused; p[1..n] = 1..n

    # ---- Helper: dot product from index k to n ----
    def _dotprod(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], k: int) -> float:
        s = 0.0
        for ii in range(k, n + 1):
            s += u[ii] * v[ii]
        return s

    # ---- Forward elimination ----
    for j in range(1, n):  # j = 1 .. n-1
        # Partial pivoting: find row with largest element in column j
        pivi = j
        max_val = abs(A[p[pivi], j])
        for ii in range(j + 1, n + 1):
            this_abs = abs(A[p[ii], j])
            if this_abs > max_val:
                max_val = this_abs
                pivi = ii
        pivrow = p[pivi]
        if pivi != j:
            p[pivi], p[j] = p[j], p[pivi]

        pivot = A[pivrow, j]
        if abs(pivot) <= ASSUMED_ZERO:
            return x, False

        for ii in range(j + 1, n + 1):
            pi = p[ii]
            mult = A[pi, j] / pivot
            if abs(mult) > ASSUMED_ZERO:
                A[pi, j] = mult
                # SubtractRow: u[i] -= m * v[i] for i = j+1..n
                for kk in range(j + 1, n + 1):
                    A[pi, kk] = A[pi, kk] - mult * A[pivrow, kk]
                b[pi] = b[pi] - mult * b[pivrow]
            else:
                A[pi, j] = 0.0

    # ---- Check last pivot ----
    pi = p[n]
    success = abs(A[pi, n]) > ASSUMED_ZERO
    if not success:
        return x, False

    # ---- Back substitution ----
    x[n] = b[pi] / A[pi, n]
    for ii in range(n - 1, 0, -1):  # i = n-1 downto 1
        pi = p[ii]
        x[ii] = (b[pi] - _dotprod(A[pi], x, ii + 1)) / A[pi, ii]

    return x, True


# ---------------------------------------------------------------------------
# Internal: SetUpEquations  (from NEWTON.PAS)
# ---------------------------------------------------------------------------

def _setup_equations(
    NO: int,
    FF: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    compounds: int,
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    stoich_matrix: npt.NDArray[np.float64],
    C0: npt.NDArray[np.float64],
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
    TracerConcIn: float | None = None,
    MassIn: npt.NDArray[np.float64] | None = None,
    FlowInTotal: npt.NDArray[np.float64] | None = None,
    FlowFromPrevious: npt.NDArray[np.float64] | None = None,
) -> None:
    """Compute the residual vector FF for the Newton solver.

    Two paths:
      - TracerEquations (compounds == 1): mass balance on a single tracer
      - ReactionEquations (compounds == 13): full ASM1 mass balance

    The choice is made by checking ``SizeJ == LastReactor + 1``, which
    is equivalent to ``compounds == 1`` in the Pascal code.

    Parameters
    ----------
    NO : int
        Function evaluation index (0 or 1) -- selects which slot of FF
        to store the result in (FF[NO, k, i]).
    FF : ndarray, shape (2, MaxReacP1+2, TotalCompounds+1)
        Residual storage array (modified in place).
    C : ndarray
        Concentration matrix.
    compounds : int
        Number of compounds (1 = tracer, 13 = full).
    plant_config : PlantConfig
        Reactor and flow parameters.
    kinetic_params, stoich_params : KineticParams, StoichiometricParams
        Model parameters.
    stoich_matrix : ndarray
        Stoichiometric matrix Stoich[compound, process].
    C0 : ndarray
        Influent concentration vector.
    air_on_hetero, air_off_hetero, air_on_auto, air_off_auto : ndarray
        DO switching functions per reactor.
    TracerConcIn : float or None
        Influent tracer concentration (only used when compounds == 1).
    MassIn : ndarray or None
        Override for influent mass flow. If None, computed from flows
        and concentrations.
    FlowInTotal : ndarray or None
        Override for total inflow per reactor. If None, uses
        plant_config.FlowInTotal.
    FlowFromPrevious : ndarray or None
        Override for flow from previous reactor. If None, uses
        plant_config.FlowFromPrevious.
    """
    last_reactor = plant_config.LastReactor
    pc = plant_config

    # Use overrides or fall back to plant_config arrays
    flow_in_total = FlowInTotal if FlowInTotal is not None else pc.FlowInTotal
    flow_from_prev = FlowFromPrevious if FlowFromPrevious is not None else pc.FlowFromPrevious

    size_j = (last_reactor + 1) * compounds

    if size_j == last_reactor + 1:
        # ---- TracerEquations ----
        tracer_conc_in = TracerConcIn if TracerConcIn is not None else 100.0
        for k in range(1, last_reactor + 2):  # 1 .. LastReactor+1
            FF[NO, k, 1] = (
                pc.FlowFeed * pc.FracFeed[k] * tracer_conc_in
                + flow_from_prev[k] * C[k - 1, 1]
                + pc.FlowRASrecycle * pc.FlagRASIn[k] * C[last_reactor + 1, 1]
                + pc.FlowArecycle * pc.FlagAIn[k] * C[pc.ReactorAOut, 1]
                + pc.FlowBrecycle * pc.FlagBIn[k] * C[pc.ReactorBOut, 1]
                - flow_in_total[k] * C[k, 1]
            )
    else:
        # ---- ReactionEquations ----
        for k in range(1, last_reactor + 1):  # 1 .. LastReactor
            # Compute process rates for reactor k
            Rho = process_rates(
                C, k, kinetic_params, stoich_params,
                air_on_hetero, air_off_hetero,
                air_on_auto, air_off_auto,
            )

            for i in range(1, compounds + 1):  # 1 .. Compounds
                mass_rate_in = (
                    flow_from_prev[k] * C[k - 1, i]
                    + pc.FlowFeed * pc.FracFeed[k] * C0[i]
                    + pc.FlowRASrecycle * C[last_reactor + 1, i] * pc.FlagRASIn[k]
                    + pc.FlowArecycle * C[pc.ReactorAOut, i] * pc.FlagAIn[k]
                    + pc.FlowBrecycle * C[pc.ReactorBOut, i] * pc.FlagBIn[k]
                )

                rate = 0.0
                for j in range(1, NO_PROCESSES + 1):
                    if stoich_matrix[i, j] != 0.0:
                        rate += stoich_matrix[i, j] * Rho[j]

                FF[NO, k, i] = mass_rate_in - flow_in_total[k] * C[k, i] + rate * pc.Vol[k]

        # ---- Settling tank (reactor LastReactor + 1) ----
        # Particulate compounds: thickening by RAS ratio
        for i in range(1, NO_PART + 1):  # 1 .. NoPart
            FF[NO, last_reactor + 1, i] = (
                (pc.FlowFeed + pc.FlowRASrecycle - pc.FlowWaste) * C[last_reactor, i]
                - pc.FlowRASrecycle * C[last_reactor + 1, i]
            )

        # Soluble compounds: pass through (same concentration)
        for i in range(NO_PART + 1, compounds + 1):  # NoPart+1 .. Compounds
            FF[NO, last_reactor + 1, i] = C[last_reactor, i] - C[last_reactor + 1, i]


# ---------------------------------------------------------------------------
# Internal: CalculateJMat  (from NEWTON.PAS)
# ---------------------------------------------------------------------------

def _calculate_jacobian(
    C: npt.NDArray[np.float64],
    FF: npt.NDArray[np.float64],
    JMat: npt.NDArray[np.float64],
    compounds: int,
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    stoich_matrix: npt.NDArray[np.float64],
    C0: npt.NDArray[np.float64],
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
    TracerConcIn: float | None = None,
    MassIn: npt.NDArray[np.float64] | None = None,
    FlowInTotal: npt.NDArray[np.float64] | None = None,
    FlowFromPrevious: npt.NDArray[np.float64] | None = None,
) -> None:
    """Compute the numerical Jacobian matrix via finite differences.

    Translates the Pascal ``CalculateJMat`` procedure from NEWTON.PAS.

    The Jacobian is stored in ``JMat`` (modified in place).
    ``FF[0, ...]`` must already contain the baseline function evaluation.
    ``FF[1, ...]`` is used as workspace for the perturbed evaluation.

    Parameters
    ----------
    C : ndarray
        Concentration matrix (modified temporarily, restored on exit).
    FF : ndarray
        Residual array (slot 0 = baseline, slot 1 = perturbed).
    JMat : ndarray, shape (MaxN+1, MaxN+1)
        Jacobian matrix (1-based, modified in place).
    compounds : int
        Number of compounds being solved.
    (remaining parameters as for _setup_equations)
    """
    last_reactor = plant_config.LastReactor

    for L in range(1, last_reactor + 2):  # L = 1 .. LastReactor+1
        for m in range(1, compounds + 1):  # m = 1 .. Compounds
            sub_j = compounds * (L - 1) + m

            # Perturbation size
            Del = np.sqrt(MACHEPS) * abs(C[L, m]) + 1.0e-08

            # Perturb
            C[L, m] += Del

            # Evaluate perturbed equations
            _setup_equations(
                1, FF, C, compounds,
                plant_config, kinetic_params, stoich_params,
                stoich_matrix, C0,
                air_on_hetero, air_off_hetero,
                air_on_auto, air_off_auto,
                TracerConcIn=TracerConcIn,
                MassIn=MassIn,
                FlowInTotal=FlowInTotal,
                FlowFromPrevious=FlowFromPrevious,
            )

            # Compute Jacobian column
            for kk in range(1, last_reactor + 2):  # kk = 1 .. LastReactor+1
                sub_i_base = compounds * (kk - 1)
                for ii in range(1, compounds + 1):  # ii = 1 .. Compounds
                    JMat[sub_i_base + ii, sub_j] = (FF[1, kk, ii] - FF[0, kk, ii]) / Del

            # Restore
            C[L, m] -= Del


# ---------------------------------------------------------------------------
# Internal: TransformFVector  (from NEWTON.PAS)
# ---------------------------------------------------------------------------

def _transform_f_vector(
    FF: npt.NDArray[np.float64],
    F: npt.NDArray[np.float64],
    compounds: int,
    last_reactor: int,
) -> None:
    """Transform residual array FF[0] into the negated flat vector F.

    Translates ``TransformFVector`` from NEWTON.PAS:
        F[Compounds*(k-1)+i] := -FF[0, k, i]
    """
    for k in range(1, last_reactor + 2):  # 1 .. LastReactor+1
        for i in range(1, compounds + 1):
            F[compounds * (k - 1) + i] = -FF[0, k, i]


# ---------------------------------------------------------------------------
# Internal: CalcNewX  (from NEWTON.PAS)
# ---------------------------------------------------------------------------

def _calc_new_x(
    C: npt.NDArray[np.float64],
    H: npt.NDArray[np.float64],
    compounds: int,
    last_reactor: int,
) -> None:
    """Apply damped Newton update to concentration matrix C.

    Translates ``CalcNewX`` from NEWTON.PAS.  The damping factor ``t``
    prevents concentrations from going negative:
        t = min(1, 0.99 * |C[k,i] / H[L]|)  when H[L] < 0
    """
    for k in range(1, last_reactor + 2):  # 1 .. LastReactor+1
        for i in range(1, compounds + 1):
            L = compounds * (k - 1) + i
            t = 1.0
            if H[L] < 0.0:
                t = min(t, 0.99 * abs(C[k, i] / H[L]))
            C[k, i] = C[k, i] + t * H[L]


# ---------------------------------------------------------------------------
# Internal: FindMaxF  (from NEWTON.PAS)
# ---------------------------------------------------------------------------

def _find_max_f(
    FF: npt.NDArray[np.float64],
    compounds: int,
    last_reactor: int,
) -> float:
    """Find the maximum absolute residual across all equations.

    Translates ``FindMaxF`` from NEWTON.PAS.
    """
    max_f = 1.0e-20
    for k in range(1, last_reactor + 2):  # 1 .. LastReactor+1
        for i in range(1, compounds + 1):
            val = abs(FF[0, k, i])
            if val > max_f:
                max_f = val
    return max_f


# ---------------------------------------------------------------------------
# Main Newton-Raphson solver  (from NEWTON.PAS)
# ---------------------------------------------------------------------------

def newton(
    C: npt.NDArray[np.float64],
    compounds: int,
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    stoich_matrix: npt.NDArray[np.float64],
    C0: npt.NDArray[np.float64],
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
    MassIn: npt.NDArray[np.float64] | None = None,
    FlowInTotal: npt.NDArray[np.float64] | None = None,
    FlowFromPrevious: npt.NDArray[np.float64] | None = None,
    TracerConcIn: float | None = None,
    max_iterations: int = 200,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Solve the nonlinear mass-balance equations using Newton-Raphson.

    Translates the main ``Newton`` procedure from NEWTON.PAS.

    Supports two modes:
      - ``compounds = 1``: tracer mode (single compound mass balance,
        used by the wastage calculation)
      - ``compounds = 13``: full 13-compound ASM1 reaction mass balance

    Parameters
    ----------
    C : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Concentration matrix, modified in place with the converged solution.
    compounds : int
        Number of compounds (1 for tracer, 13 for full).
    plant_config : PlantConfig
        Reactor topology and flow parameters.
    kinetic_params : KineticParams
        Temperature-adjusted kinetic constants.
    stoich_params : StoichiometricParams
        Stoichiometric/yield parameters.
    stoich_matrix : ndarray
        Stoichiometric matrix Stoich[compound, process].
    C0 : ndarray
        Influent concentration vector.
    air_on_hetero, air_off_hetero : ndarray
        Heterotrophic DO switching functions per reactor (1-based).
    air_on_auto, air_off_auto : ndarray
        Autotrophic DO switching functions per reactor (1-based).
    MassIn : ndarray or None
        Optional override for influent mass flow.
    FlowInTotal : ndarray or None
        Optional override for total inflow per reactor.
    FlowFromPrevious : ndarray or None
        Optional override for flow from previous reactor.
    TracerConcIn : float or None
        Influent tracer concentration (only used when compounds == 1).
    max_iterations : int
        Maximum number of Newton iterations (safety limit, not in Pascal
        original which uses an unconditional Repeat..Until).

    Returns
    -------
    C : ndarray
        Updated concentration matrix (same object as input).
    converged : bool
        True if ``MaxF < 1e-03`` was achieved within ``max_iterations``.
    """
    last_reactor = plant_config.LastReactor
    size_j = (last_reactor + 1) * compounds

    # Allocate working arrays
    # FF[NO, k, i] -- residual storage (NO=0: baseline, NO=1: perturbed)
    FF = np.zeros((2, MAX_REAC_P1 + 2, TOTAL_COMPOUNDS + 1), dtype=np.float64)
    # JMat[1..SizeJ, 1..SizeJ] -- Jacobian matrix
    JMat = np.zeros((size_j + 1, size_j + 1), dtype=np.float64)
    # F[1..SizeJ] -- negated residual vector
    F = np.zeros(size_j + 1, dtype=np.float64)
    # H[1..SizeJ] -- Newton step vector
    H = np.zeros(size_j + 1, dtype=np.float64)

    # Common keyword args for equation setup
    eq_kwargs = dict(
        compounds=compounds,
        plant_config=plant_config,
        kinetic_params=kinetic_params,
        stoich_params=stoich_params,
        stoich_matrix=stoich_matrix,
        C0=C0,
        air_on_hetero=air_on_hetero,
        air_off_hetero=air_off_hetero,
        air_on_auto=air_on_auto,
        air_off_auto=air_off_auto,
        TracerConcIn=TracerConcIn,
        MassIn=MassIn,
        FlowInTotal=FlowInTotal,
        FlowFromPrevious=FlowFromPrevious,
    )

    # Initial function evaluation
    max_f = 1.0e10
    _setup_equations(0, FF, C, **eq_kwargs)

    inversions = 0
    converged = False

    while True:
        # Calculate Jacobian (uses FF[0] as baseline, fills FF[1])
        _calculate_jacobian(C, FF, JMat, **eq_kwargs)

        # Transform F vector: F[L] = -FF[0, k, i]
        _transform_f_vector(FF, F, compounds, last_reactor)

        # Solve J * H = F via Gauss elimination
        # Make copies since gauss modifies A and b in place
        JMat_copy = JMat.copy()
        F_copy = F.copy()
        H, okay = gauss(size_j, JMat_copy, F_copy)

        if not okay:
            return C, False

        # Damped Newton update
        _calc_new_x(C, H, compounds, last_reactor)

        # Re-evaluate equations at new point
        _setup_equations(0, FF, C, **eq_kwargs)

        # Check convergence
        max_f = _find_max_f(FF, compounds, last_reactor)

        inversions += 1

        if max_f < 1.0e-03:
            converged = True
            break

        if inversions >= max_iterations:
            break

        # Reset JMat for next iteration
        JMat[:] = 0.0

    return C, converged
