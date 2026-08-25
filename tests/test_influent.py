"""Tests for the influent wastewater fractionation module."""

import numpy as np
import pytest

from uct_activated_sludge.constants import ASM1Component as Comp, TOTAL_COMPOUNDS
from uct_activated_sludge.influent import (
    fractionate_influent,
    raw_wastewater_params,
    settled_wastewater_params,
)
from uct_activated_sludge.models import StoichiometricParams, WastewaterParams


class TestFractionateInfluent:
    """Tests for fractionate_influent."""

    def test_c0_shape(self, default_ww_params, default_stoich_params):
        """C0 should have shape (TOTAL_COMPOUNDS + 1,) for 1-based indexing."""
        C0 = fractionate_influent(default_ww_params, default_stoich_params)
        assert C0.shape == (TOTAL_COMPOUNDS + 1,)

    def test_all_values_ge_zero(self, default_ww_params, default_stoich_params):
        """All C0 values (indices 1..13) should be >= 0 (negative values are clamped to 1e-06)."""
        C0 = fractionate_influent(default_ww_params, default_stoich_params)
        for i in range(1, TOTAL_COMPOUNDS):  # 1..13
            assert C0[i] >= 0.0, f"C0[{i}] = {C0[i]} is negative"

    def test_negative_values_are_clamped(self):
        """If a fractionation formula produces a negative value, it should be clamped to 1e-06."""
        # Create params where OrgN would go negative by setting high Fup and low Nti
        ww = WastewaterParams(Sti=500.0, Nti=5.0, Fup=0.40, Fus=0.05, Fxbh=0.0,
                              Fnaa=0.10, Fnox=0.50, Fnu=0.03)
        sp = StoichiometricParams()
        C0 = fractionate_influent(ww, sp)
        # All values should still be >= 1e-06 (clamped)
        for i in range(1, TOTAL_COMPOUNDS):
            assert C0[i] >= 1e-06 or C0[i] == 0.0, f"C0[{i}] = {C0[i]} is negative"

    def test_do_is_zero(self, default_ww_params, default_stoich_params):
        """C0[14] (dissolved oxygen) should be zero."""
        C0 = fractionate_influent(default_ww_params, default_stoich_params)
        assert C0[Comp.DO] == 0.0

    def test_index_zero_is_zero(self, default_ww_params, default_stoich_params):
        """C0[0] (unused padding) should be zero."""
        C0 = fractionate_influent(default_ww_params, default_stoich_params)
        assert C0[0] == 0.0

    def test_raw_wastewater_fractionation_values(self):
        """Test specific C0 values for raw wastewater with default params."""
        ww = WastewaterParams()  # raw defaults
        sp = StoichiometricParams()
        C0 = fractionate_influent(ww, sp)

        # C0[1] = Fxbh * Sti = 0.0 * 500 = 0.0 (zero, not negative, so not clamped)
        np.testing.assert_allclose(C0[Comp.Xbh], 0.0, atol=1e-15)

        # C0[8] = Fbs * (1 - Fus - Fup) * Sti = 0.20 * (1 - 0.05 - 0.13) * 500 = 82.0
        expected_ss = 0.20 * (1.0 - 0.05 - 0.13) * 500.0
        np.testing.assert_allclose(C0[Comp.Ss], expected_ss, rtol=1e-10)

        # C0[13] = Fus * Sti = 0.05 * 500 = 25.0
        np.testing.assert_allclose(C0[Comp.Si], 0.05 * 500.0, rtol=1e-10)

        # C0[9] = Fnaa * Nti = 0.75 * 50 = 37.5
        np.testing.assert_allclose(C0[Comp.Snh], 0.75 * 50.0, rtol=1e-10)

        # C0[11] = 0.044 (small nitrate seed)
        np.testing.assert_allclose(C0[Comp.Sno], 0.044, rtol=1e-10)

        # C0[12] = Alki = 10.0
        np.testing.assert_allclose(C0[Comp.Alk], 10.0, rtol=1e-10)

        # C0[5] = (1 - Fbs) * (1 - Fus - Fup - Fxbh) * Sti
        expected_sbp = (1.0 - 0.20) * (1.0 - 0.05 - 0.13 - 0.0) * 500.0
        np.testing.assert_allclose(C0[Comp.Sbp], expected_sbp, rtol=1e-10)

        # C0[6] = Fup * Sti = 0.13 * 500 = 65.0
        np.testing.assert_allclose(C0[Comp.Xi], 0.13 * 500.0, rtol=1e-10)

    def test_settled_wastewater_gives_different_fractionation(self):
        """Settled wastewater should produce different C0 values than raw."""
        ww_raw = WastewaterParams.raw()
        ww_settled = WastewaterParams.settled_sewage()
        sp = StoichiometricParams()

        C0_raw = fractionate_influent(ww_raw, sp)
        C0_settled = fractionate_influent(ww_settled, sp)

        # Settled sewage has different Fbs, Fus, Fup so various compounds differ
        # Ss: Fbs differs (0.20 vs 0.25)
        assert C0_raw[Comp.Ss] != C0_settled[Comp.Ss]
        # Si: Fus differs (0.05 vs 0.08)
        assert C0_raw[Comp.Si] != C0_settled[Comp.Si]
        # Sbp: depends on Fbs and Fup
        assert C0_raw[Comp.Sbp] != C0_settled[Comp.Sbp]

    def test_stoich_params_default(self):
        """When stoich_params is None, default StoichiometricParams should be used."""
        ww = WastewaterParams()
        C0_with_none = fractionate_influent(ww, None)
        C0_with_default = fractionate_influent(ww, StoichiometricParams())
        np.testing.assert_array_equal(C0_with_none, C0_with_default)

    def test_custom_sti_and_nti(self):
        """Custom Sti and Nti should scale the outputs proportionally."""
        ww1 = WastewaterParams(Sti=500.0, Nti=50.0)
        ww2 = WastewaterParams(Sti=1000.0, Nti=100.0)
        sp = StoichiometricParams()

        C0_1 = fractionate_influent(ww1, sp)
        C0_2 = fractionate_influent(ww2, sp)

        # Ss should scale with Sti
        np.testing.assert_allclose(C0_2[Comp.Ss], 2.0 * C0_1[Comp.Ss], rtol=1e-10)
        # Snh should scale with Nti
        np.testing.assert_allclose(C0_2[Comp.Snh], 2.0 * C0_1[Comp.Snh], rtol=1e-10)
        # Si should scale with Sti
        np.testing.assert_allclose(C0_2[Comp.Si], 2.0 * C0_1[Comp.Si], rtol=1e-10)


class TestRawWastewaterParams:
    """Tests for raw_wastewater_params convenience function."""

    def test_returns_wastewater_params(self):
        """Should return a WastewaterParams instance."""
        ww = raw_wastewater_params()
        assert isinstance(ww, WastewaterParams)

    def test_default_fractionation(self):
        """Default raw fractionation should match WATER.PAS Raw preset."""
        ww = raw_wastewater_params()
        assert ww.Fbs == 0.20
        assert ww.Fus == 0.05
        assert ww.Fup == 0.13
        assert ww.Fnaa == 0.75
        assert ww.settled is False

    def test_custom_sti(self):
        """Custom Sti should be passed through."""
        ww = raw_wastewater_params(Sti=800.0)
        assert ww.Sti == 800.0


class TestSettledWastewaterParams:
    """Tests for settled_wastewater_params convenience function."""

    def test_returns_wastewater_params(self):
        """Should return a WastewaterParams instance."""
        ww = settled_wastewater_params()
        assert isinstance(ww, WastewaterParams)

    def test_settled_fractionation(self):
        """Settled fractionation should match WATER.PAS Settled preset."""
        ww = settled_wastewater_params()
        assert ww.Fbs == 0.25
        assert ww.Fus == 0.08
        assert ww.Fup == 0.04
        assert ww.Fnaa == 0.83
        assert ww.settled is True
