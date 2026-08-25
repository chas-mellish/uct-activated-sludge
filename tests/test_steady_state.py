"""Tier 2 integration tests for the steady-state simulation driver."""

import numpy as np
import pytest

from uct_activated_sludge.models import (
    KineticParams,
    PlantConfig,
    StoichiometricParams,
    WastewaterParams,
)
from uct_activated_sludge.steady_state import run_steady_state


def _make_plant_config():
    """Create a fully configured 3-reactor UCT plant for steady-state testing."""
    pc = PlantConfig()
    pc.LastReactor = 3

    pc.Vol[1] = 1.5
    pc.Vol[2] = 3.0
    pc.Vol[3] = 6.0

    pc.FracFeed[1] = 1.0

    pc.DOConc[1] = 0.0
    pc.DOConc[2] = 0.0
    pc.DOConc[3] = 2.0

    pc.ReactorAerated[1] = False
    pc.ReactorAerated[2] = False
    pc.ReactorAerated[3] = True

    pc.FlowFeed = 25.0
    pc.FlowRASrecycle = 25.0
    pc.FlowArecycle = 75.0
    pc.FlowBrecycle = 0.0

    pc.FlagRASIn[1] = 1
    pc.FlagAIn[2] = 1
    pc.FlagAOut[3] = 1
    pc.ReactorAIn = 2
    pc.ReactorAOut = 3

    pc.Rs = 20.0
    pc.Temp = 20.0
    pc.VolumeTotal = 10.5

    return pc


class TestRunSteadyState:
    """Integration tests for run_steady_state."""

    @pytest.fixture
    def steady_state_result(self):
        """Run a steady-state simulation and return the result dict."""
        pc = _make_plant_config()
        kp = KineticParams()
        sp = StoichiometricParams()
        ww = WastewaterParams()

        return run_steady_state(pc, kp, sp, ww)

    def test_converges(self, steady_state_result):
        """The Newton solver should converge for this standard configuration."""
        assert steady_state_result["converged"] is True

    def test_result_contains_expected_keys(self, steady_state_result):
        """The result dict should contain all documented keys."""
        expected_keys = {
            "C", "C0", "CSteady", "Stoich",
            "Oc", "On", "Ot", "Denit",
            "air_on_hetero", "air_off_hetero",
            "air_on_auto", "air_off_auto",
            "kinetic_params", "plant_config",
            "converged", "output", "FlowWaste",
        }
        assert expected_keys.issubset(steady_state_result.keys())

    def test_concentrations_are_positive(self, steady_state_result):
        """Converged concentrations should be non-negative for active reactors."""
        C = steady_state_result["C"]
        pc = steady_state_result["plant_config"]

        for k in range(1, pc.LastReactor + 1):
            for i in range(1, 14):  # compounds 1..13
                assert C[k, i] >= 0.0, (
                    f"C[{k},{i}] = {C[k, i]} is negative"
                )

    def test_concentrations_are_finite(self, steady_state_result):
        """All concentrations should be finite (no NaN or inf)."""
        C = steady_state_result["C"]
        pc = steady_state_result["plant_config"]

        for k in range(1, pc.LastReactor + 2):
            for i in range(1, 14):
                assert np.isfinite(C[k, i]), (
                    f"C[{k},{i}] = {C[k, i]} is not finite"
                )

    def test_flow_waste_is_positive(self, steady_state_result):
        """Wastage flow should be positive."""
        assert steady_state_result["FlowWaste"] > 0.0

    def test_output_is_string(self, steady_state_result):
        """Output report should be a non-empty string."""
        output = steady_state_result["output"]
        assert isinstance(output, str)
        assert len(output) > 0
        assert "STEADY STATE" in output

    def test_our_rates_nonnegative_in_aerobic(self, steady_state_result):
        """OUR rates should be non-negative in the aerobic reactor."""
        Oc = steady_state_result["Oc"]
        On = steady_state_result["On"]
        Ot = steady_state_result["Ot"]

        # Reactor 3 is aerobic
        assert Oc[3] >= 0.0
        assert On[3] >= 0.0
        assert Ot[3] >= 0.0

    def test_biomass_positive_in_all_reactors(self, steady_state_result):
        """Heterotrophic biomass (Xbh) should be positive in all reactors."""
        C = steady_state_result["C"]
        for k in range(1, 4):
            assert C[k, 1] > 0.0, f"Xbh in reactor {k} is not positive"

    def test_csteady_matches_c(self, steady_state_result):
        """CSteady should be a copy of the converged C for active reactors."""
        C = steady_state_result["C"]
        CSteady = steady_state_result["CSteady"]
        pc = steady_state_result["plant_config"]

        for k in range(1, pc.LastReactor + 1):
            for i in range(1, 14):
                np.testing.assert_allclose(
                    CSteady[k, i], C[k, i], rtol=1e-12,
                    err_msg=f"CSteady[{k},{i}] != C[{k},{i}]",
                )

    def test_validation_errors(self):
        """run_steady_state should raise ValueError for invalid configurations."""
        kp = KineticParams()
        sp = StoichiometricParams()
        ww = WastewaterParams()

        # Rs = 0
        pc = _make_plant_config()
        pc.Rs = 0.0
        with pytest.raises(ValueError, match="Rs"):
            run_steady_state(pc, kp, sp, ww)

        # FlowFeed = 0
        pc = _make_plant_config()
        pc.FlowFeed = 0.0
        with pytest.raises(ValueError, match="FlowFeed"):
            run_steady_state(pc, kp, sp, ww)

        # VolumeTotal = 0
        pc = _make_plant_config()
        pc.VolumeTotal = 0.0
        with pytest.raises(ValueError, match="VolumeTotal"):
            run_steady_state(pc, kp, sp, ww)

        # FlowRASrecycle = 0
        pc = _make_plant_config()
        pc.FlowRASrecycle = 0.0
        with pytest.raises(ValueError, match="FlowRASrecycle"):
            run_steady_state(pc, kp, sp, ww)
