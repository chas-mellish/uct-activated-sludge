"""Influent wastewater fractionation into ASM1 component vector.

Ported from the Pascal source files:
  - FRACINF.PAS  -- FractionateInfluent procedure
  - WATER.PAS    -- Raw and Settled sewage presets

The fractionation converts bulk wastewater parameters (Sti, Nti, and
the various ``F*`` fractions) into the 13-component influent vector
``C0[1..13]`` used by the ASM1 model.

All arrays use 1-based indexing (index 0 is unused padding) to match
the Pascal original.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from uct_activated_sludge.constants import ASM1Component, TOTAL_COMPOUNDS
from uct_activated_sludge.models import StoichiometricParams, WastewaterParams

__all__ = [
    "fractionate_influent",
    "raw_wastewater_params",
    "settled_wastewater_params",
]

# Minimum clamp value for negative concentrations (matches Pascal original)
_CLAMP_MIN: float = 1.0e-06


def fractionate_influent(
    ww_params: WastewaterParams,
    stoich_params: StoichiometricParams | None = None,
) -> npt.NDArray[np.float64]:
    """Convert bulk wastewater parameters into the ASM1 influent vector.

    This is a direct port of the ``FractionateInfluent`` procedure from
    FRACINF.PAS.  The procedure splits the total influent COD (``Sti``)
    and TKN (``Nti``) into the 13 ASM1 state variables using the
    fractionation parameters stored in *ww_params*.

    Parameters
    ----------
    ww_params : WastewaterParams
        Influent concentrations and fractionation parameters.
    stoich_params : StoichiometricParams, optional
        Stoichiometric coefficients; only ``Ixb`` and ``Ixe`` are used
        (nitrogen content of biomass and endogenous residue).  If *None*,
        defaults are constructed from :class:`StoichiometricParams`.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(TOTAL_COMPOUNDS + 1,)`` with 1-based
        indexing.  ``C0[0]`` is unused padding.  ``C0[1]`` through
        ``C0[13]`` hold the 13 ASM1 state-variable influent concentrations.
        ``C0[14]`` (dissolved oxygen) is zero.
        All values in ``C0[1..13]`` are clamped to >= 1e-06.

    Notes
    -----
    The fractionation formulas exactly replicate FRACINF.PAS:

    .. code-block:: pascal

        C0[1]  := Fxbh * Sti;
        C0[5]  := (1 - Fbs) * (1 - Fus - Fup - Fxbh) * Sti;
        C0[6]  := Fup * Sti;
        OrgN   := (1 - Fnaa - Fnu) * Nti - Ixe * Fup * Sti - Ixb * Fxbh * Sti;
        -- if OrgN < 0 then adjust Fnaa and set OrgN = 0 --
        C0[7]  := OrgN * Fnox;
        C0[10] := OrgN * (1 - Fnox);
        C0[8]  := Fbs * (1 - Fus - Fup) * Sti;
        C0[9]  := Fnaa * Nti;
        C0[11] := 0.044;        -- small NO3 seed --
        C0[12] := Alki;
        C0[13] := Fus * Sti;
    """
    if stoich_params is None:
        stoich_params = StoichiometricParams()

    # Unpack wastewater fractionation parameters
    Sti = ww_params.Sti
    Nti = ww_params.Nti
    Fbs = ww_params.Fbs
    Fus = ww_params.Fus
    Fup = ww_params.Fup
    Fnaa = ww_params.Fnaa
    Fnox = ww_params.Fnox
    Fnu = ww_params.Fnu
    Fxbh = ww_params.Fxbh
    Alki = ww_params.Alki

    # Unpack stoichiometric N-content fractions
    Ixb = stoich_params.Ixb
    Ixe = stoich_params.Ixe

    # Initialise C0 vector to zero (1-based: indices 0..TOTAL_COMPOUNDS)
    C0 = np.zeros(TOTAL_COMPOUNDS + 1, dtype=np.float64)

    # --- COD fractionation ---
    # C0[1] = Xbh: heterotrophic biomass COD in influent
    C0[ASM1Component.Xbh] = Fxbh * Sti

    # C0[5] = Sbp: particulate biodegradable COD
    C0[ASM1Component.Sbp] = (1.0 - Fbs) * (1.0 - Fus - Fup - Fxbh) * Sti

    # C0[6] = Xi: particulate unbiodegradable COD
    C0[ASM1Component.Xi] = Fup * Sti

    # --- Nitrogen fractionation ---
    # Organic nitrogen = total N minus ammonia-N minus unbiodegradable-N
    #                     minus N bound in influent unbiodeg. particulates
    #                     minus N bound in influent biomass
    OrgN = (1.0 - Fnaa - Fnu) * Nti - Ixe * Fup * Sti - Ixb * Fxbh * Sti

    if OrgN < 0:
        # Adjust Fnaa upward so OrgN becomes exactly zero
        # (matches Pascal: Fnaa := (Nti - Fnu*Nti - Ixe*Fup*Sti - Ixb*Fxbh*Sti)/Nti)
        Fnaa = (Nti - Fnu * Nti - Ixe * Fup * Sti - Ixb * Fxbh * Sti) / Nti
        OrgN = 0.0

    # C0[7] = Xnd: particulate biodegradable organic N
    C0[ASM1Component.Xnd] = OrgN * Fnox

    # C0[10] = Snd: soluble organic N
    C0[ASM1Component.Snd] = OrgN * (1.0 - Fnox)

    # C0[8] = Ss: soluble biodegradable COD
    C0[ASM1Component.Ss] = Fbs * (1.0 - Fus - Fup) * Sti

    # C0[9] = Snh: ammonia nitrogen (free + saline)
    C0[ASM1Component.Snh] = Fnaa * Nti

    # C0[11] = Sno: small nitrate seed for cases when Rs < Rsm
    C0[ASM1Component.Sno] = 0.044

    # C0[12] = Alk: influent alkalinity
    C0[ASM1Component.Alk] = Alki

    # C0[13] = Si: soluble unbiodegradable COD
    C0[ASM1Component.Si] = Fus * Sti

    # --- Clamp negative values to 1e-06 (Pascal: if C0[i] < 0 then 1e-06) ---
    for i in range(1, TOTAL_COMPOUNDS + 1):
        if C0[i] < 0:
            C0[i] = _CLAMP_MIN

    return C0


# -----------------------------------------------------------------------
# Preset helpers  (from WATER.PAS Raw / Settled procedures)
# -----------------------------------------------------------------------

def raw_wastewater_params(
    Sti: float = 500.0,
    Nti: float = 50.0,
    Alki: float = 10.0,
) -> WastewaterParams:
    """Return a :class:`WastewaterParams` with **raw** sewage fractionation.

    Mirrors the ``Raw`` procedure in WATER.PAS::

        Fbs    := 0.20;   Fus    := 0.05;   Fup    := 0.13;
        Fnaa   := 0.75;   Fnox   := 0.50;   Fnu    := 0.03;
        Fxbh   := 0.0;    VSSTSS := 0.75;

    Parameters
    ----------
    Sti : float
        Influent total COD (g COD m-3), default 500.
    Nti : float
        Influent total TKN (g N m-3), default 50.
    Alki : float
        Influent alkalinity (mmol/L as CaCO3), default 10.
    """
    return WastewaterParams.raw(Sti=Sti, Nti=Nti, Alki=Alki)


def settled_wastewater_params(
    Sti: float = 500.0,
    Nti: float = 50.0,
    Alki: float = 10.0,
) -> WastewaterParams:
    """Return a :class:`WastewaterParams` with **settled** sewage fractionation.

    Mirrors the ``Settled`` procedure in WATER.PAS::

        Fbs    := 0.25;   Fus    := 0.08;   Fup    := 0.04;
        Fnaa   := 0.83;   Fnox   := 0.50;   Fnu    := 0.04;
        Fxbh   := 0.0;    VSSTSS := 0.83;

    Parameters
    ----------
    Sti : float
        Influent total COD (g COD m-3), default 500.
    Nti : float
        Influent total TKN (g N m-3), default 50.
    Alki : float
        Influent alkalinity (mmol/L as CaCO3), default 10.
    """
    return WastewaterParams.settled_sewage(Sti=Sti, Nti=Nti, Alki=Alki)
