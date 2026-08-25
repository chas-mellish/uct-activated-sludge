"""Tests for the two-timescale adaptive predictor-corrector ODE integrator."""

import numpy as np
import pytest

from uct_activated_sludge.models import IntegrationParams


class TestIntegrationParameters:
    """Tests for the integration_parameters function."""

    def test_sets_step_sizes(self):
        """integration_parameters should set DeltaTlarge and DeltaTsmall."""
        from uct_activated_sludge.integrator import integration_parameters

        ip = IntegrationParams()
        result = integration_parameters(ip, data_int_hours=0.25)

        # DeltaTlarge = 0.25 / 60 / 24 = 0.000173611...
        expected_large = 0.25 / 60.0 / 24.0
        np.testing.assert_allclose(result.DeltaTlarge, expected_large, rtol=1e-12)

        # DeltaTsmall = DeltaTlarge / 5
        np.testing.assert_allclose(result.DeltaTsmall, expected_large / 5.0, rtol=1e-12)

    def test_sets_integ_int_days(self):
        """IntegIntDays should be data_int_hours / 24."""
        from uct_activated_sludge.integrator import integration_parameters

        ip = IntegrationParams()
        result = integration_parameters(ip, data_int_hours=2.0)

        np.testing.assert_allclose(result.IntegIntDays, 2.0 / 24.0, rtol=1e-12)

    def test_sets_max_bounds(self):
        """MaxDeltaTLarge should equal IntegIntDays, MaxDeltaTSmall equals DeltaTlarge."""
        from uct_activated_sludge.integrator import integration_parameters

        ip = IntegrationParams()
        result = integration_parameters(ip, data_int_hours=0.5)

        np.testing.assert_allclose(result.MaxDeltaTLarge, 0.5 / 24.0, rtol=1e-12)
        np.testing.assert_allclose(
            result.MaxDeltaTSmall, 0.5 / 60.0 / 24.0, rtol=1e-12
        )

    def test_sets_min_bounds(self):
        """MinDeltaTLarge should be DeltaTlarge/1000, MinDeltaTSmall DeltaTsmall/1000."""
        from uct_activated_sludge.integrator import integration_parameters

        ip = IntegrationParams()
        result = integration_parameters(ip, data_int_hours=0.25)

        expected_large = 0.25 / 60.0 / 24.0
        expected_small = expected_large / 5.0
        np.testing.assert_allclose(result.MinDeltaTLarge, expected_large / 1000.0, rtol=1e-12)
        np.testing.assert_allclose(result.MinDeltaTSmall, expected_small / 1000.0, rtol=1e-12)

    def test_returns_same_object(self):
        """integration_parameters modifies in place and returns the same object."""
        from uct_activated_sludge.integrator import integration_parameters

        ip = IntegrationParams()
        result = integration_parameters(ip, data_int_hours=0.25)
        assert result is ip

    def test_resets_truncation_state(self):
        """Truncation flags should be reset to False."""
        from uct_activated_sludge.integrator import integration_parameters

        ip = IntegrationParams()
        ip.TruncatedLarge = True
        ip.TruncatedSmall = True
        ip.PrevDeltaTLarge = 999.0
        ip.PrevDeltaTSmall = 999.0

        result = integration_parameters(ip, data_int_hours=0.25)
        assert result.TruncatedLarge is False
        assert result.TruncatedSmall is False
        assert result.PrevDeltaTLarge == 0.0
        assert result.PrevDeltaTSmall == 0.0

    def test_sets_data_int_hours(self):
        """DataIntHours should be stored in the params."""
        from uct_activated_sludge.integrator import integration_parameters

        ip = IntegrationParams()
        result = integration_parameters(ip, data_int_hours=1.5)
        assert result.DataIntHours == 1.5


class TestIntegratorImport:
    """Basic import tests for the integrator module."""

    def test_integrate_function_exists(self):
        """The integrate function should be importable."""
        from uct_activated_sludge.integrator import integrate
        assert callable(integrate)

    def test_integration_parameters_function_exists(self):
        """The integration_parameters function should be importable."""
        from uct_activated_sludge.integrator import integration_parameters
        assert callable(integration_parameters)


class TestIntegrateExecution:
    """Integration tests that call integrate() with realistic reactor state."""

    @pytest.fixture
    def steady_state_context(self):
        """Run a steady-state solve and prepare all inputs for integrate()."""
        from uct_activated_sludge.steady_state import run_steady_state
        from uct_activated_sludge.models import (
            KineticParams,
            PlantConfig,
            StoichiometricParams,
            WastewaterParams,
        )
        from uct_activated_sludge.hydraulics import flow_division_dynamic
        from uct_activated_sludge.influent import fractionate_influent
        from uct_activated_sludge.integrator import integration_parameters, integrate
        from uct_activated_sludge.kinetics import air_supply

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

        kp = KineticParams()
        sp = StoichiometricParams()
        ww = WastewaterParams()

        ss = run_steady_state(pc, kp, sp, ww)

        return ss

    def test_integrate_preserves_finite_concentrations(self, steady_state_context):
        """After one integration interval, concentrations should remain finite."""
        from uct_activated_sludge.integrator import integrate, integration_parameters
        from uct_activated_sludge.hydraulics import flow_division_dynamic
        from uct_activated_sludge.influent import fractionate_influent
        from uct_activated_sludge.kinetics import air_supply
        from uct_activated_sludge.models import (
            IntegrationParams,
            StoichiometricParams,
            WastewaterParams,
        )
        from uct_activated_sludge.constants import MAX_REAC_P1, TOTAL_COMPOUNDS

        ss = steady_state_context
        pc = ss["plant_config"]
        kp = ss["kinetic_params"]
        sp = StoichiometricParams()
        ww = WastewaterParams()

        # Use steady-state concentrations as initial state
        C = ss["CSteady"].copy()

        # Set up integration parameters
        ip = IntegrationParams()
        data_int_hours = 0.25
        integration_parameters(ip, data_int_hours)

        # Get influent fractionation
        C0 = fractionate_influent(ww, sp)

        # Get air supply switching functions
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, kp)

        # Compute dynamic flow division
        wastage_rate = pc.FlowWaste
        flow_div = flow_division_dynamic(
            last_reactor=pc.LastReactor,
            flow_feed=pc.FlowFeed,
            flow_ras_recycle=pc.FlowRASrecycle,
            flow_a_recycle=pc.FlowArecycle,
            flow_b_recycle=pc.FlowBrecycle,
            wastage_rate=wastage_rate,
            frac_feed=pc.FracFeed,
            vol=pc.Vol,
            flag_ras_in=pc.FlagRASIn,
            flag_a_in=pc.FlagAIn,
            flag_b_in=pc.FlagBIn,
            flag_a_out=pc.FlagAOut,
            flag_b_out=pc.FlagBOut,
        )

        # Allocate workspace
        shape_rc = (MAX_REAC_P1 + 1, TOTAL_COMPOUNDS + 1)
        MassRateIn = np.zeros(shape_rc, dtype=np.float64)

        # Call integrate
        result = integrate(
            C,
            pc,
            kp,
            sp,
            ip,
            ss["Stoich"],
            C0,
            MassRateIn,
            flow_div["DhOut"],
            flow_div["DhFeed"],
            flow_div["DhRAS"],
            flow_div["DhFromPrevious"],
            flow_div["DhArecycle"],
            flow_div["DhBrecycle"],
            air_on_h, air_off_h, air_on_a, air_off_a,
            data_int_hours,
        )

        # All concentrations should be finite after integration
        for k in range(1, pc.LastReactor + 1):
            for i in range(1, 14):
                assert np.isfinite(C[k, i]), (
                    f"C[{k},{i}] is not finite after integrate(): {C[k, i]}"
                )

    def test_integrate_concentrations_non_negative(self, steady_state_context):
        """After one integration interval, concentrations should be non-negative."""
        from uct_activated_sludge.integrator import integrate, integration_parameters
        from uct_activated_sludge.hydraulics import flow_division_dynamic
        from uct_activated_sludge.influent import fractionate_influent
        from uct_activated_sludge.kinetics import air_supply
        from uct_activated_sludge.models import (
            IntegrationParams,
            StoichiometricParams,
            WastewaterParams,
        )
        from uct_activated_sludge.constants import MAX_REAC_P1, TOTAL_COMPOUNDS

        ss = steady_state_context
        pc = ss["plant_config"]
        kp = ss["kinetic_params"]
        sp = StoichiometricParams()
        ww = WastewaterParams()

        C = ss["CSteady"].copy()

        ip = IntegrationParams()
        data_int_hours = 0.25
        integration_parameters(ip, data_int_hours)

        C0 = fractionate_influent(ww, sp)
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, kp)

        wastage_rate = pc.FlowWaste
        flow_div = flow_division_dynamic(
            last_reactor=pc.LastReactor,
            flow_feed=pc.FlowFeed,
            flow_ras_recycle=pc.FlowRASrecycle,
            flow_a_recycle=pc.FlowArecycle,
            flow_b_recycle=pc.FlowBrecycle,
            wastage_rate=wastage_rate,
            frac_feed=pc.FracFeed,
            vol=pc.Vol,
            flag_ras_in=pc.FlagRASIn,
            flag_a_in=pc.FlagAIn,
            flag_b_in=pc.FlagBIn,
            flag_a_out=pc.FlagAOut,
            flag_b_out=pc.FlagBOut,
        )

        shape_rc = (MAX_REAC_P1 + 1, TOTAL_COMPOUNDS + 1)
        MassRateIn = np.zeros(shape_rc, dtype=np.float64)

        integrate(
            C,
            pc,
            kp,
            sp,
            ip,
            ss["Stoich"],
            C0,
            MassRateIn,
            flow_div["DhOut"],
            flow_div["DhFeed"],
            flow_div["DhRAS"],
            flow_div["DhFromPrevious"],
            flow_div["DhArecycle"],
            flow_div["DhBrecycle"],
            air_on_h, air_off_h, air_on_a, air_off_a,
            data_int_hours,
        )

        # Concentrations should remain non-negative (within numerical tolerance)
        for k in range(1, pc.LastReactor + 1):
            for i in range(1, 14):
                assert C[k, i] >= -1e-6, (
                    f"C[{k},{i}] is significantly negative after integrate(): {C[k, i]}"
                )
