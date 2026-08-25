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
