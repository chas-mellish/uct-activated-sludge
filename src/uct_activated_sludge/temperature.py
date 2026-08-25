"""Arrhenius temperature correction for kinetic parameters.

Ported from ADJTEMP.PAS — the ``TempAdjustment`` procedure.

Each kinetic rate constant is adjusted from its 20 degC reference value
using the Arrhenius equation:

    K(T) = K(20) * theta ** (T - 20)

where *theta* is the parameter-specific temperature-correction factor.
"""

from __future__ import annotations

from dataclasses import replace

from uct_activated_sludge.models import KineticParams


def adjust_temperature(params: KineticParams, temp: float) -> KineticParams:
    """Return a new :class:`KineticParams` with temperature-adjusted values.

    Parameters
    ----------
    params:
        Kinetic parameters containing 20 degC reference values and
        Arrhenius theta coefficients.
    temp:
        Operating temperature in degrees Celsius.

    Returns
    -------
    KineticParams
        A **new** dataclass instance.  The 20 degC reference values and
        theta coefficients are unchanged; only the working (non-``20``)
        fields are updated.

    Notes
    -----
    Mirrors the Pascal procedure ``TempAdjustment`` in ADJTEMP.PAS:

    .. code-block:: pascal

        TempDiff := Temp - 20.0;
        MuHatHetero := MuHatHetero20 * Power(ThetaMuHatH, TempDiff);
        Ks          := Ks20          * Power(ThetaKs,      TempDiff);
        Bh          := Bh20          * Power(ThetaBh,      TempDiff);
        Kmp         := Kmp20         * Power(ThetaKmp,     TempDiff);
        Ksp         := Ksp20         * Power(ThetaKsp,     TempDiff);
        Ka          := Ka20          * Power(ThetaKa,      TempDiff);
        Kr          := Kr20          * Power(ThetaKr,      TempDiff);
        MuHatAuto   := MuHatAuto20   * Power(ThetaMuHatA,  TempDiff);
        Knh         := Knh20         * Power(ThetaKnh,     TempDiff);
        Ba          := Ba20          * Power(ThetaBa,      TempDiff);

    ``Power(B, X)`` in Pascal is ``B ** X`` in Python.
    """
    td = temp - 20.0

    return replace(
        params,
        MuHatHetero=params.MuHatHetero20 * params.ThetaMuHatH ** td,
        Ks=params.Ks20 * params.ThetaKs ** td,
        Bh=params.Bh20 * params.ThetaBh ** td,
        Kmp=params.Kmp20 * params.ThetaKmp ** td,
        Ksp=params.Ksp20 * params.ThetaKsp ** td,
        Ka=params.Ka20 * params.ThetaKa ** td,
        Kr=params.Kr20 * params.ThetaKr ** td,
        MuHatAuto=params.MuHatAuto20 * params.ThetaMuHatA ** td,
        Knh=params.Knh20 * params.ThetaKnh ** td,
        Ba=params.Ba20 * params.ThetaBa ** td,
    )
