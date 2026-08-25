"""ASM1 kinetic rate expressions, air-supply switching, and utilization rates.

Ported from the original Pascal source files:
  - RATEUNIT.PAS  -- ProcessRates procedure (14 process rate expressions)
  - RATEUNIT.PAS  -- AirSupply procedure (DO switching functions)
  - RATEUNIT.PAS  -- UtilizationRates procedure (OUR and denitrification)

All arrays use 1-based indexing (index 0 is unused padding) to match
the Pascal original and simplify cross-validation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from uct_activated_sludge.constants import (
    ASM1Component as Comp,
    ASM1Process as Proc,
    NO_PROCESSES,
)
from uct_activated_sludge.models import (
    KineticParams,
    PlantConfig,
    StoichiometricParams,
    _zeros_1based,
)

__all__ = [
    "process_rates",
    "air_supply",
    "utilization_rates",
]


# ---------------------------------------------------------------------------
# 1. ProcessRates  (from RATEUNIT.PAS)
# ---------------------------------------------------------------------------

def process_rates(
    C: npt.NDArray[np.float64],
    k: int,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the 14 ASM1 process rates for reactor *k*.

    Parameters
    ----------
    C : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Concentration matrix.  ``C[k, i]`` is compound *i* in reactor *k*.
    k : int
        1-based reactor index.
    kinetic_params : KineticParams
        Temperature-adjusted kinetic constants.
    stoich_params : StoichiometricParams
        Stoichiometric/yield parameters (only ``Fma`` is used here).
    air_on_hetero, air_off_hetero : ndarray
        Heterotrophic DO switching functions per reactor (1-based).
    air_on_auto, air_off_auto : ndarray
        Autotrophic DO switching functions per reactor (1-based).

    Returns
    -------
    Rho : ndarray, shape (NO_PROCESSES + 1,)
        1-based process-rate vector.  ``Rho[j]`` is the rate of process *j*.
    """
    kp = kinetic_params
    sp = stoich_params

    Rho = np.zeros(NO_PROCESSES + 1, dtype=np.float64)

    # -- Intermediate ("dummy") terms --
    # MuHeteroS: heterotrophic specific growth rate on soluble substrate Ss
    #   MuHatHetero * C[k, Xbh] * C[k, Ss] / (Ks + C[k, Ss])
    MuHeteroS = kp.MuHatHetero * C[k, Comp.Xbh] * C[k, Comp.Ss] / (kp.Ks + C[k, Comp.Ss])

    # MuHeteroP: heterotrophic specific growth rate on stored substrate Xs
    #   Kmp * C[k, Xs] / (Ksp + C[k, Xs] / C[k, Xbh])
    # Guard: when Xbh is zero there is no biomass so the rate is zero.
    if C[k, Comp.Xbh] == 0.0:
        MuHeteroP = 0.0
    else:
        MuHeteroP = kp.Kmp * C[k, Comp.Xs] / (kp.Ksp + C[k, Comp.Xs] / C[k, Comp.Xbh])

    # Monod switching terms for nitrate and ammonia
    NO3Limit = C[k, Comp.Sno] / (kp.Kno + C[k, Comp.Sno])
    NH3Limit = C[k, Comp.Snh] / (kp.Kna + C[k, Comp.Snh])

    # -- Process 1: Aerobic growth of heterotrophs on Ss with NH3 --
    Rho[Proc.AerobicGrowthHeteroSsNH3] = (
        MuHeteroS * air_on_hetero[k] * NH3Limit
    )

    # -- Process 2: Aerobic growth of heterotrophs on Ss with NO3 --
    Rho[Proc.AerobicGrowthHeteroSsNO3] = (
        MuHeteroS * air_on_hetero[k] * (1.0 - NH3Limit) * NO3Limit
    )

    # -- Process 3: Anoxic growth of heterotrophs on Ss with NH3 --
    Rho[Proc.AnoxicGrowthHeteroSsNH3] = (
        MuHeteroS * air_off_hetero[k] * NO3Limit * NH3Limit
    )

    # -- Process 4: Anoxic growth of heterotrophs on Ss with NO3 --
    Rho[Proc.AnoxicGrowthHeteroSsNO3] = (
        MuHeteroS * air_off_hetero[k] * NO3Limit * (1.0 - NH3Limit)
    )

    # -- Process 5: Aerobic growth of heterotrophs on Xs with NH3 --
    Rho[Proc.AerobicGrowthHeteroXsNH3] = (
        MuHeteroP * air_on_hetero[k] * NH3Limit
    )

    # -- Process 6: Aerobic growth of heterotrophs on Xs with NO3 --
    Rho[Proc.AerobicGrowthHeteroXsNO3] = (
        MuHeteroP * air_on_hetero[k] * NO3Limit * (1.0 - NH3Limit)
    )

    # -- Process 7: Anoxic growth of heterotrophs on Xs with NH3 --
    Rho[Proc.AnoxicGrowthHeteroXsNH3] = (
        MuHeteroP * kp.NetaGrow * air_off_hetero[k] * NO3Limit * NH3Limit
    )

    # -- Process 8: Anoxic growth of heterotrophs on Xs with NO3 --
    Rho[Proc.AnoxicGrowthHeteroXsNO3] = (
        MuHeteroP * kp.NetaGrow * air_off_hetero[k] * NO3Limit * (1.0 - NH3Limit)
    )

    # -- Process 9: Decay of heterotrophs --
    Rho[Proc.DecayHeterotrophs] = kp.Bh * C[k, Comp.Xbh]

    # -- Process 10: Storage of particulate influent COD --
    # Guard: when Xbh is zero the storage rate is zero.
    if C[k, Comp.Xbh] == 0.0:
        Rho[Proc.StorageParticulateCOD] = 0.0
    else:
        Rho[Proc.StorageParticulateCOD] = (
            kp.Ka * C[k, Comp.Sbp] * C[k, Comp.Xbh]
            * (sp.Fma - C[k, Comp.Xs] / C[k, Comp.Xbh])
        )

    # -- Process 11: Hydrolysis of biodegradable particulate organic N --
    if C[k, Comp.Xs] == 0.0:
        Rho[Proc.HydrolysisPartOrgN] = 0.0
    else:
        Rho[Proc.HydrolysisPartOrgN] = (
            (Rho[Proc.AerobicGrowthHeteroXsNH3]
             + Rho[Proc.AerobicGrowthHeteroXsNO3]
             + Rho[Proc.AnoxicGrowthHeteroXsNH3]
             + Rho[Proc.AnoxicGrowthHeteroXsNO3])
            * C[k, Comp.Xnd] / C[k, Comp.Xs]
        )

    # -- Process 12: Ammonification of soluble organic N --
    Rho[Proc.AmmonificationSolOrgN] = kp.Kr * C[k, Comp.Snd] * C[k, Comp.Xbh]

    # -- Process 13: Autotrophic growth --
    Rho[Proc.AutotrophicGrowth] = (
        kp.MuHatAuto * C[k, Comp.Xba]
        * (C[k, Comp.Snh] / (kp.Knh + C[k, Comp.Snh]))
        * air_on_auto[k]
    )

    # -- Process 14: Autotrophic decay --
    Rho[Proc.AutotrophicDecay] = kp.Ba * C[k, Comp.Xba]

    return Rho


# ---------------------------------------------------------------------------
# 2. AirSupply  (from RATEUNIT.PAS)
# ---------------------------------------------------------------------------

def air_supply(
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute DO switching functions for all reactors.

    Parameters
    ----------
    plant_config : PlantConfig
        Must contain ``LastReactor`` and ``DOConc[k]`` for each reactor.
    kinetic_params : KineticParams
        Must contain ``Koh`` (heterotrophic half-sat for DO) and
        ``Koa`` (autotrophic half-sat for DO).

    Returns
    -------
    air_on_hetero : ndarray
        ``DOConc[k] / (Koh + DOConc[k])`` for each reactor (1-based).
    air_off_hetero : ndarray
        ``Koh / (Koh + DOConc[k])`` for each reactor (1-based).
    air_on_auto : ndarray
        ``DOConc[k] / (Koa + DOConc[k])`` for each reactor (1-based).
    air_off_auto : ndarray
        ``Koa / (Koa + DOConc[k])`` for each reactor (1-based).
    """
    from uct_activated_sludge.models import MAX_REAC_P1

    air_on_hetero = _zeros_1based(MAX_REAC_P1)
    air_off_hetero = _zeros_1based(MAX_REAC_P1)
    air_on_auto = _zeros_1based(MAX_REAC_P1)
    air_off_auto = _zeros_1based(MAX_REAC_P1)

    Koh = kinetic_params.Koh
    Koa = kinetic_params.Koa

    for k in range(1, plant_config.LastReactor + 1):
        do_k = plant_config.DOConc[k]
        air_on_hetero[k] = do_k / (Koh + do_k)
        air_off_hetero[k] = Koh / (Koh + do_k)
        air_on_auto[k] = do_k / (Koa + do_k)
        air_off_auto[k] = Koa / (Koa + do_k)

    return air_on_hetero, air_off_hetero, air_on_auto, air_off_auto


# ---------------------------------------------------------------------------
# 3. UtilizationRates  (from RATEUNIT.PAS)
# ---------------------------------------------------------------------------

def utilization_rates(
    C: npt.NDArray[np.float64],
    Stoich: npt.NDArray[np.float64],
    plant_config: PlantConfig,
    kinetic_params: KineticParams,
    stoich_params: StoichiometricParams,
    air_on_hetero: npt.NDArray[np.float64],
    air_off_hetero: npt.NDArray[np.float64],
    air_on_auto: npt.NDArray[np.float64],
    air_off_auto: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute per-reactor oxygen-uptake and denitrification rates.

    In the Pascal original, ``UtilizationRates`` calls ``ProcessRates``
    internally for each reactor.  This Python version does the same.

    Parameters
    ----------
    C : ndarray, shape (MaxReacP1+1, TotalCompounds+1)
        Concentration matrix.
    Stoich : ndarray, shape (TotalCompounds+1, NoProcesses+1)
        Stoichiometric matrix.  ``Stoich[i, j]`` is the stoichiometric
        coefficient of compound *i* in process *j*.
    plant_config : PlantConfig
        Reactor topology (``LastReactor``).
    kinetic_params : KineticParams
        Temperature-adjusted kinetic constants.
    stoich_params : StoichiometricParams
        Stoichiometric/yield parameters.
    air_on_hetero, air_off_hetero : ndarray
        Heterotrophic DO switching functions (1-based).
    air_on_auto, air_off_auto : ndarray
        Autotrophic DO switching functions (1-based).

    Returns
    -------
    Oc : ndarray
        Carbonaceous OUR per reactor (g O2 m-3 h-1), 1-based.
    On : ndarray
        Nitrogenous OUR per reactor (g O2 m-3 h-1), 1-based.
    Ot : ndarray
        Total OUR per reactor (g O2 m-3 h-1), 1-based.
    Denit : ndarray
        Denitrification rate per reactor (g N m-3 h-1), 1-based.
    """
    from uct_activated_sludge.models import MAX_REAC_P1

    Oc = _zeros_1based(MAX_REAC_P1)
    On = _zeros_1based(MAX_REAC_P1)
    Ot = _zeros_1based(MAX_REAC_P1)
    Denit = _zeros_1based(MAX_REAC_P1)

    # Compound index for DO row in Stoich matrix
    DO = Comp.DO      # 14
    Sno = Comp.Sno    # 11

    for k in range(1, plant_config.LastReactor + 1):
        Rho = process_rates(
            C, k, kinetic_params, stoich_params,
            air_on_hetero, air_off_hetero,
            air_on_auto, air_off_auto,
        )

        # Carbonaceous OUR:
        #   Oc[k] = -(Stoich[14,1]*Rho[1] + Stoich[14,2]*Rho[2]
        #             + Stoich[14,5]*Rho[5] + Stoich[14,6]*Rho[6]) / 24.0
        Oc[k] = -(
            Stoich[DO, Proc.AerobicGrowthHeteroSsNH3] * Rho[Proc.AerobicGrowthHeteroSsNH3]
            + Stoich[DO, Proc.AerobicGrowthHeteroSsNO3] * Rho[Proc.AerobicGrowthHeteroSsNO3]
            + Stoich[DO, Proc.AerobicGrowthHeteroXsNH3] * Rho[Proc.AerobicGrowthHeteroXsNH3]
            + Stoich[DO, Proc.AerobicGrowthHeteroXsNO3] * Rho[Proc.AerobicGrowthHeteroXsNO3]
        ) / 24.0

        # Nitrogenous OUR:
        #   On[k] = -Stoich[14,13]*Rho[13] / 24.0
        On[k] = -(
            Stoich[DO, Proc.AutotrophicGrowth] * Rho[Proc.AutotrophicGrowth]
        ) / 24.0

        # Total OUR:
        Ot[k] = Oc[k] + On[k]

        # Denitrification rate:
        #   Denit[k] = -(Stoich[11,2]*Rho[2] + Stoich[11,3]*Rho[3]
        #                + Stoich[11,4]*Rho[4] + Stoich[11,6]*Rho[6]
        #                + Stoich[11,7]*Rho[7] + Stoich[11,8]*Rho[8]) / 24.0
        Denit[k] = -(
            Stoich[Sno, Proc.AerobicGrowthHeteroSsNO3] * Rho[Proc.AerobicGrowthHeteroSsNO3]
            + Stoich[Sno, Proc.AnoxicGrowthHeteroSsNH3] * Rho[Proc.AnoxicGrowthHeteroSsNH3]
            + Stoich[Sno, Proc.AnoxicGrowthHeteroSsNO3] * Rho[Proc.AnoxicGrowthHeteroSsNO3]
            + Stoich[Sno, Proc.AerobicGrowthHeteroXsNO3] * Rho[Proc.AerobicGrowthHeteroXsNO3]
            + Stoich[Sno, Proc.AnoxicGrowthHeteroXsNH3] * Rho[Proc.AnoxicGrowthHeteroXsNH3]
            + Stoich[Sno, Proc.AnoxicGrowthHeteroXsNO3] * Rho[Proc.AnoxicGrowthHeteroXsNO3]
        ) / 24.0

    return Oc, On, Ot, Denit
