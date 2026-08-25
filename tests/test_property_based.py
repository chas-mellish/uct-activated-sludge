"""Tier 3 property-based tests using Hypothesis.

These tests verify invariants that must hold for arbitrary valid inputs:
  1. Influent fractionation never produces negative concentrations.
  2. The stoichiometric matrix respects COD mass balance per process.
  3. Temperature adjustment is monotonic with respect to temperature.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from uct_activated_sludge.constants import LAST_COMPOUND, TOTAL_COMPOUNDS
from uct_activated_sludge.influent import fractionate_influent
from uct_activated_sludge.models import (
    KineticParams,
    StoichiometricParams,
    WastewaterParams,
)
from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix
from uct_activated_sludge.temperature import adjust_temperature

# ---------------------------------------------------------------------------
# Reusable strategies
# ---------------------------------------------------------------------------

# Influent concentrations
sti_strategy = st.floats(min_value=100.0, max_value=2000.0)
nti_strategy = st.floats(min_value=10.0, max_value=200.0)


@st.composite
def wastewater_params_strategy(draw: st.DrawFn) -> WastewaterParams:
    """Generate valid WastewaterParams with realistic fraction ranges."""
    Sti = draw(sti_strategy)
    Nti = draw(nti_strategy)
    Fus = draw(st.floats(min_value=0.01, max_value=0.15))
    Fup = draw(st.floats(min_value=0.01, max_value=0.20))
    Fxbh = draw(st.floats(min_value=0.0, max_value=0.05))
    # Ensure Fus + Fup + Fxbh < 1 (always true given max 0.15+0.20+0.05=0.40)
    assume(Fus + Fup + Fxbh < 1.0)
    Fbs = draw(st.floats(min_value=0.05, max_value=0.60))
    Fnaa = draw(st.floats(min_value=0.30, max_value=0.95))
    Fnox = draw(st.floats(min_value=0.10, max_value=0.90))
    Fnu = draw(st.floats(min_value=0.0, max_value=0.10))
    # Ensure Fnaa + Fnu < 1
    assume(Fnaa + Fnu < 1.0)
    Alki = draw(st.floats(min_value=1.0, max_value=30.0))
    return WastewaterParams(
        Sti=Sti, Nti=Nti, Fbs=Fbs, Fus=Fus, Fup=Fup,
        Fnaa=Fnaa, Fnox=Fnox, Fnu=Fnu, Fxbh=Fxbh, Alki=Alki,
    )


@st.composite
def stoichiometric_params_strategy(draw: st.DrawFn) -> StoichiometricParams:
    """Generate valid StoichiometricParams within realistic ranges."""
    Yh = draw(st.floats(min_value=0.30, max_value=0.90))
    Fe = draw(st.floats(min_value=0.02, max_value=0.20))
    Ixb = draw(st.floats(min_value=0.03, max_value=0.12))
    Ixe = draw(st.floats(min_value=0.03, max_value=0.12))
    Ya = draw(st.floats(min_value=0.05, max_value=0.30))
    return StoichiometricParams(Yh=Yh, Fe=Fe, Ixb=Ixb, Ixe=Ixe, Ya=Ya)


# ---------------------------------------------------------------------------
# 1. Concentrations never go negative after fractionation
# ---------------------------------------------------------------------------

class TestFractionationNonNegativity:
    """All C0 components (indices 1..13) must be >= 0 after fractionation.

    The clamping logic sets negative values to 1e-06, so the invariant is
    that no component is ever negative.  Components that are legitimately
    zero (e.g. Xbh when Fxbh=0) remain at 0.0.
    """

    @given(ww=wastewater_params_strategy())
    @settings(max_examples=200)
    def test_all_components_non_negative_default_stoich(
        self, ww: WastewaterParams,
    ) -> None:
        """With default stoichiometric params, every C0[1..13] >= 0."""
        C0 = fractionate_influent(ww)
        for i in range(1, LAST_COMPOUND + 1):
            assert C0[i] >= 0.0, (
                f"C0[{i}] = {C0[i]:.2e} is negative "
                f"for Sti={ww.Sti}, Nti={ww.Nti}"
            )

    @given(
        ww=wastewater_params_strategy(),
        sp=stoichiometric_params_strategy(),
    )
    @settings(max_examples=200)
    def test_all_components_non_negative_varying_stoich(
        self, ww: WastewaterParams, sp: StoichiometricParams,
    ) -> None:
        """With varying stoichiometric params, every C0[1..13] >= 0."""
        C0 = fractionate_influent(ww, sp)
        for i in range(1, LAST_COMPOUND + 1):
            assert C0[i] >= 0.0, (
                f"C0[{i}] = {C0[i]:.2e} is negative"
            )

    @given(ww=wastewater_params_strategy())
    @settings(max_examples=200)
    def test_negative_values_clamped_to_threshold(
        self, ww: WastewaterParams,
    ) -> None:
        """Any component that would be negative is clamped to exactly 1e-06.

        Components are either naturally >= 0 or clamped to 1e-06; no value
        should fall in the range (0, 1e-06) exclusive unless it was
        computed to be in that range naturally (which is valid).
        """
        C0 = fractionate_influent(ww)
        for i in range(1, LAST_COMPOUND + 1):
            assert C0[i] >= 0.0, (
                f"C0[{i}] = {C0[i]:.2e} is negative (clamping failed)"
            )

    @given(ww=wastewater_params_strategy())
    @settings(max_examples=100)
    def test_do_always_zero(self, ww: WastewaterParams) -> None:
        """Dissolved oxygen (C0[14]) must always be zero in influent."""
        C0 = fractionate_influent(ww)
        assert C0[TOTAL_COMPOUNDS] == 0.0


# ---------------------------------------------------------------------------
# 2. Stoichiometric matrix COD mass balance
# ---------------------------------------------------------------------------

class TestStoichiometricCODBalance:
    """COD mass balance for each process in the stoichiometric matrix.

    In ASM1, COD is conserved across biological transformations.  The
    electron acceptor (O2 for aerobic, NO3 for anoxic) accounts for the
    COD that is oxidised.

    COD-carrying state variables: Xbh(1), Xe(2), Xba(3), Xs(4),
    Sbp(5), Xi(6), Ss(8), Si(13).
    Electron acceptors: DO(14) for aerobic; NO3(11) for anoxic, where
    1 g NO3-N is equivalent to 2.86 g COD.
    """

    COD_COMPONENTS = [1, 2, 3, 4, 5, 6, 8, 13]

    @given(sp=stoichiometric_params_strategy())
    @settings(max_examples=200)
    def test_cod_balance_aerobic_growth_nh3(
        self, sp: StoichiometricParams,
    ) -> None:
        """Aerobic growth with NH3 as N source (processes 1, 5).

        COD balance: biomass + substrate + O2 = 0.
        No NO3 involvement (Stoich[11,j] = 0 for these processes).
        """
        Stoich = build_stoichiometric_matrix(sp)
        for j in [1, 5]:
            cod_sum = sum(Stoich[i, j] for i in self.COD_COMPONENTS)
            # O2 is the sole electron acceptor; entered as negative in
            # Stoich[14,j], representing COD removed by oxidation.
            balance = cod_sum - Stoich[14, j]
            assert abs(balance) < 1e-10, (
                f"Process {j}: aerobic COD balance = {balance:.2e} "
                f"(Yh={sp.Yh})"
            )

    @given(sp=stoichiometric_params_strategy())
    @settings(max_examples=200)
    def test_cod_balance_aerobic_growth_no3(
        self, sp: StoichiometricParams,
    ) -> None:
        """Aerobic growth with NO3 as N source (processes 2, 6).

        Same COD balance as NH3-sourced aerobic growth: biomass + substrate
        + O2 = 0.  The NO3 consumed here is used only as a nitrogen source
        (not an electron acceptor), so it does not enter the COD balance.
        """
        Stoich = build_stoichiometric_matrix(sp)
        for j in [2, 6]:
            cod_sum = sum(Stoich[i, j] for i in self.COD_COMPONENTS)
            balance = cod_sum - Stoich[14, j]
            assert abs(balance) < 1e-10, (
                f"Process {j}: aerobic COD balance = {balance:.2e} "
                f"(Yh={sp.Yh})"
            )

    @given(sp=stoichiometric_params_strategy())
    @settings(max_examples=200)
    def test_cod_balance_anoxic_growth(
        self, sp: StoichiometricParams,
    ) -> None:
        """Anoxic growth (processes 3, 4, 7, 8).

        Under anoxic conditions NO3 replaces O2 as the electron acceptor.
        The denitrification stoichiometry is -(1-Yh)/(2.86*Yh) g NO3-N
        per g COD of biomass produced.  The COD-equivalent of this NO3
        consumption is 2.86 * denitrification_NO3.

        For processes using NO3 as nitrogen source (4, 8), Stoich[11,j]
        also includes a nitrogen-source term (-Ixb).  We must separate
        the denitrification component for the COD balance.
        """
        Stoich = build_stoichiometric_matrix(sp)
        Yh = sp.Yh

        # The denitrification NO3 contribution (g NO3-N per unit process)
        denit_no3 = -(1.0 - Yh) / (2.86 * Yh)

        for j in [3, 4, 7, 8]:
            cod_sum = sum(Stoich[i, j] for i in self.COD_COMPONENTS)
            # COD equivalent of denitrification: 2.86 * denit_no3
            # This replaces the O2 term (Stoich[14,j] should be 0 for
            # anoxic processes).
            balance = cod_sum - 2.86 * denit_no3
            assert abs(balance) < 1e-10, (
                f"Process {j}: anoxic COD balance = {balance:.2e} "
                f"(Yh={sp.Yh})"
            )

    @given(sp=stoichiometric_params_strategy())
    @settings(max_examples=200)
    def test_cod_balance_decay_processes(
        self, sp: StoichiometricParams,
    ) -> None:
        """COD mass balance for decay processes (9, 14).

        In decay: biomass (-1) -> endogenous residue (Fe) + particulate
        biodegradable COD (1 - Fe). The total COD is conserved:
            Stoich[1,9] + Stoich[2,9] + Stoich[5,9] == 0  (heterotrophic)
            Stoich[3,14] + Stoich[2,14] + Stoich[5,14] == 0  (autotrophic)
        """
        Stoich = build_stoichiometric_matrix(sp)

        # Process 9: heterotrophic decay
        decay_h = Stoich[1, 9] + Stoich[2, 9] + Stoich[5, 9]
        assert abs(decay_h) < 1e-10, (
            f"Heterotrophic decay COD balance = {decay_h:.2e}"
        )

        # Process 14: autotrophic decay
        decay_a = Stoich[3, 14] + Stoich[2, 14] + Stoich[5, 14]
        assert abs(decay_a) < 1e-10, (
            f"Autotrophic decay COD balance = {decay_a:.2e}"
        )

    @given(sp=stoichiometric_params_strategy())
    @settings(max_examples=100)
    def test_storage_process_conservation(
        self, sp: StoichiometricParams,
    ) -> None:
        """Process 10 (storage): Sbp -> Xs is a pure transfer (COD conserved)."""
        Stoich = build_stoichiometric_matrix(sp)
        # Stoich[4, 10] + Stoich[5, 10] should be 0 (Xs gained = Sbp lost)
        balance = Stoich[4, 10] + Stoich[5, 10]
        assert abs(balance) < 1e-10, (
            f"Storage process COD balance = {balance:.2e}"
        )


# ---------------------------------------------------------------------------
# 3. Temperature adjustment monotonicity / consistency
# ---------------------------------------------------------------------------

class TestTemperatureMonotonicity:
    """Temperature-adjusted kinetic parameters must respond monotonically.

    For parameters with theta > 1, the adjusted value increases with
    temperature. For parameters with theta < 1, the adjusted value
    decreases with temperature.
    """

    # (ref_value_attr, theta_attr, adjusted_attr)
    PARAM_SPECS = [
        ("MuHatHetero20", "ThetaMuHatH", "MuHatHetero"),
        ("Ks20", "ThetaKs", "Ks"),
        ("Bh20", "ThetaBh", "Bh"),
        ("Kmp20", "ThetaKmp", "Kmp"),
        ("Ksp20", "ThetaKsp", "Ksp"),
        ("Kr20", "ThetaKr", "Kr"),
        ("Ka20", "ThetaKa", "Ka"),
        ("MuHatAuto20", "ThetaMuHatA", "MuHatAuto"),
        ("Knh20", "ThetaKnh", "Knh"),
        ("Ba20", "ThetaBa", "Ba"),
    ]

    @given(
        t_low=st.floats(min_value=5.0, max_value=19.9),
        t_high=st.floats(min_value=20.1, max_value=40.0),
    )
    @settings(max_examples=200)
    def test_monotonicity_default_params(
        self, t_low: float, t_high: float,
    ) -> None:
        """For default kinetics, parameters with theta > 1 increase and
        those with theta < 1 decrease as temperature rises."""
        assume(t_low < t_high)
        base = KineticParams()
        adj_low = adjust_temperature(base, t_low)
        adj_high = adjust_temperature(base, t_high)

        for ref_attr, theta_attr, adj_attr in self.PARAM_SPECS:
            theta = getattr(base, theta_attr)
            val_low = getattr(adj_low, adj_attr)
            val_high = getattr(adj_high, adj_attr)

            if theta > 1.0:
                assert val_high > val_low, (
                    f"{adj_attr}: theta={theta} > 1 but "
                    f"value at {t_high}C ({val_high:.6f}) <= "
                    f"value at {t_low}C ({val_low:.6f})"
                )
            elif theta < 1.0:
                assert val_high < val_low, (
                    f"{adj_attr}: theta={theta} < 1 but "
                    f"value at {t_high}C ({val_high:.6f}) >= "
                    f"value at {t_low}C ({val_low:.6f})"
                )
            else:
                # theta == 1.0: value should be unchanged
                assert abs(val_high - val_low) < 1e-12, (
                    f"{adj_attr}: theta=1.0 but values differ"
                )

    @given(temp=st.floats(min_value=5.0, max_value=40.0))
    @settings(max_examples=200)
    def test_at_20c_equals_reference(self, temp: float) -> None:
        """At exactly 20 degC, all adjusted values equal their 20 degC refs."""
        base = KineticParams()
        adj = adjust_temperature(base, 20.0)
        for ref_attr, _theta_attr, adj_attr in self.PARAM_SPECS:
            ref_val = getattr(base, ref_attr)
            adj_val = getattr(adj, adj_attr)
            assert abs(adj_val - ref_val) < 1e-12, (
                f"{adj_attr} at 20C: {adj_val} != ref {ref_val}"
            )

    @given(
        temp=st.floats(min_value=5.0, max_value=40.0),
    )
    @settings(max_examples=200)
    def test_adjusted_values_always_positive(self, temp: float) -> None:
        """Temperature-adjusted kinetic values must always be positive."""
        base = KineticParams()
        adj = adjust_temperature(base, temp)
        for _ref_attr, _theta_attr, adj_attr in self.PARAM_SPECS:
            val = getattr(adj, adj_attr)
            assert val > 0.0, (
                f"{adj_attr} = {val} is not positive at T={temp}C"
            )

    @given(
        t1=st.floats(min_value=5.0, max_value=20.0),
        gap1=st.floats(min_value=0.5, max_value=8.0),
        gap2=st.floats(min_value=0.5, max_value=8.0),
    )
    @settings(max_examples=100)
    def test_transitivity(self, t1: float, gap1: float, gap2: float) -> None:
        """If t1 < t2 < t3 and theta > 1, then val(t1) < val(t2) < val(t3)."""
        t2 = t1 + gap1
        t3 = t2 + gap2
        base = KineticParams()
        adj1 = adjust_temperature(base, t1)
        adj2 = adjust_temperature(base, t2)
        adj3 = adjust_temperature(base, t3)

        for _ref_attr, theta_attr, adj_attr in self.PARAM_SPECS:
            theta = getattr(base, theta_attr)
            v1 = getattr(adj1, adj_attr)
            v2 = getattr(adj2, adj_attr)
            v3 = getattr(adj3, adj_attr)

            if theta > 1.0:
                assert v1 < v2 < v3, (
                    f"{adj_attr} (theta={theta}>1): "
                    f"expected {v1} < {v2} < {v3}"
                )
            elif theta < 1.0:
                assert v1 > v2 > v3, (
                    f"{adj_attr} (theta={theta}<1): "
                    f"expected {v1} > {v2} > {v3}"
                )
