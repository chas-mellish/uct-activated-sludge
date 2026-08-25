"""ASM1 component/process enumerations and domain constants.

Ported from the original Pascal source files:
  - VARSUNIT.PAS  -- domain constants and type definitions
  - STRINGS.PAS   -- compound names, units, short names
  - STCHUNIT.PAS  -- stoichiometric process definitions
  - RATEUNIT.PAS  -- process rate definitions

All arrays use 1-based indexing (index 0 is unused padding) to match
the Pascal original and simplify cross-validation.
"""

from enum import IntEnum

__all__ = [
    # Enums
    "ASM1Component",
    "ASM1Process",
    # Partition boundaries
    "FIRST_SLOW",
    "LAST_SLOW",
    "FIRST_RAPID",
    "LAST_RAPID",
    # Sizing constants
    "TOTAL_COMPOUNDS",
    "LAST_COMPOUND",
    "NO_PART",
    "NO_PROCESSES",
    "MAX_REAC",
    "MAX_REAC_P1",
    "MAX_N",
    "NO_DIURNAL_INTS",
    "NO_DI_VARS",
    # Name / unit lookup arrays
    "COMPOUND_NAMES",
    "COMPOUND_UNITS",
    "COMPOUND_SHORT_NAMES",
]


# ---------------------------------------------------------------------------
# ASM1 component enum (1-based, matching Pascal compound numbering)
# ---------------------------------------------------------------------------

class ASM1Component(IntEnum):
    """ASM1 state-variable (compound) indices.

    Indices 1-7 are slow (particulate) variables;
    indices 8-13 are rapid (soluble) variables;
    index 14 is dissolved oxygen.
    """

    Xbh = 1   # Heterotrophic biomass
    Xe = 2    # Endogenous residue
    Xba = 3   # Autotrophic biomass
    Xs = 4    # Stored COD
    Sbp = 5   # Particulate biodegradable COD
    Xi = 6    # Particulate unbiodegradable COD
    Xnd = 7   # Particulate biodegradable N
    Ss = 8    # Soluble biodegradable COD
    Snh = 9   # Ammonia N
    Snd = 10  # Soluble organic N
    Sno = 11  # Nitrate N
    Alk = 12  # Alkalinity
    Si = 13   # Soluble unbiodegradable COD
    DO = 14   # Dissolved oxygen


# ---------------------------------------------------------------------------
# ASM1 process enum (1-based, matching Pascal Rho[]/Stoich[,] indexing)
# ---------------------------------------------------------------------------

class ASM1Process(IntEnum):
    """ASM1 biological process indices.

    Processes 1-8 are growth reactions, 9 is decay, 10 is storage,
    11-12 are nitrogen transformations, and 13-14 are autotrophic.
    """

    AerobicGrowthHeteroSsNH3 = 1   # Aerobic growth of heterotrophs on Ss with NH3
    AerobicGrowthHeteroSsNO3 = 2   # Aerobic growth of heterotrophs on Ss with NO3
    AnoxicGrowthHeteroSsNH3 = 3    # Anoxic growth of heterotrophs on Ss with NH3
    AnoxicGrowthHeteroSsNO3 = 4    # Anoxic growth of heterotrophs on Ss with NO3
    AerobicGrowthHeteroXsNH3 = 5   # Aerobic growth of heterotrophs on Xs with NH3
    AerobicGrowthHeteroXsNO3 = 6   # Aerobic growth of heterotrophs on Xs with NO3
    AnoxicGrowthHeteroXsNH3 = 7    # Anoxic growth of heterotrophs on Xs with NH3
    AnoxicGrowthHeteroXsNO3 = 8    # Anoxic growth of heterotrophs on Xs with NO3
    DecayHeterotrophs = 9          # Decay of heterotrophs
    StorageParticulateCOD = 10     # Storage of particulate influent COD
    HydrolysisPartOrgN = 11        # Hydrolysis of biodegradable particulate organic N
    AmmonificationSolOrgN = 12     # Ammonification of soluble organic N
    AutotrophicGrowth = 13         # Autotrophic growth
    AutotrophicDecay = 14          # Autotrophic decay


# ---------------------------------------------------------------------------
# Domain constants  (from VARSUNIT.PAS)
# ---------------------------------------------------------------------------

# Partition boundaries for slow (particulate) vs rapid (soluble) variables
FIRST_SLOW: int = 1
LAST_SLOW: int = 7
FIRST_RAPID: int = 8
LAST_RAPID: int = 13

# Sizing constants
NO_PART: int = 7              # Number of particulate compounds
TOTAL_COMPOUNDS: int = 14     # 13 state variables + DO
LAST_COMPOUND: int = 13       # Last state variable index (excluding DO)
NO_PROCESSES: int = 14        # Number of biological processes
MAX_REAC: int = 12            # Maximum number of reactors
MAX_REAC_P1: int = 13         # MaxReac + 1
MAX_N: int = 169              # LastCompound * (MaxReac + 1)
NO_DIURNAL_INTS: int = 12     # Number of diurnal intervals
NO_DI_VARS: int = 18          # Number of diurnal output variables


# ---------------------------------------------------------------------------
# Compound name / unit / short-name lookup arrays  (from STRINGS.PAS)
#
# Index 0 is unused padding so that arr[compound_index] works directly.
# Indices 1-13 correspond to the 13 ASM1 state variables.
# Index 14 is dissolved oxygen (DO) -- not given a compound name in
# the original Pascal but included here for completeness.
# Indices 15-18 are derived output variables (Oc, On, Ot, VSS, TKN)
# that appear in the units/short-name arrays only.
# ---------------------------------------------------------------------------

COMPOUND_NAMES: list[str] = [
    "",                          # 0  (unused padding)
    "Xbh (hetero.)",             # 1
    "Xe (endog.)",               # 2
    "Xba (autotrophs)",          # 3
    "Xs (stored COD)",           # 4
    "Sbp (prt bio COD)",         # 5
    "Xi (prt unb COD)",          # 6
    "Xnd (prt bio N)",           # 7
    "Ss (sol bio COD)",          # 8
    "Snh (ammonia N)",           # 9
    "Snd (sol org N)",           # 10
    "Sno (nitrate N)",           # 11
    "Alkalinity",                # 12
    "Si (sol unb COD)",          # 13
    "DO (dissolved O2)",         # 14
]

COMPOUND_UNITS: list[str] = [
    "",                # 0  (unused padding)
    "g COD m-3",       # 1
    "g COD m-3",       # 2
    "g COD m-3",       # 3
    "g COD m-3",       # 4
    "g COD m-3",       # 5
    "g COD m-3",       # 6
    "g N m-3",         # 7
    "g COD m-3",       # 8
    "g N m-3",         # 9
    "g N m-3",         # 10
    "g N m-3",         # 11
    "mole m-3",        # 12
    "g COD m-3",       # 13
    "gO/m3/h",         # 14
    "gO/m3/h",         # 15
    "gO/m3/h",         # 16
    "g VSS m-3",       # 17
    "g N m-3",         # 18
]

COMPOUND_SHORT_NAMES: list[str] = [
    "",      # 0  (unused padding)
    "Xbh",   # 1
    "Xe",    # 2
    "Xba",   # 3
    "Xs",    # 4
    "Sbp",   # 5
    "Xi",    # 6
    "Xnd",   # 7
    "Ss",    # 8
    "Snh",   # 9
    "Snd",   # 10
    "Sno",   # 11
    "Alk",   # 12
    "Si",    # 13
    "Oc",    # 14
    "On",    # 15
    "Ot",    # 16
    "VSS",   # 17
    "TKN",   # 18
]
