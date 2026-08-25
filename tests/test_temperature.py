"""Tests for the Arrhenius temperature correction module."""

import numpy as np
import pytest

from uct_activated_sludge.models import KineticParams
from uct_activated_sludge.temperature import adjust_temperature


class TestAdjustTemperature:
    """Tests for adjust_temperature."""

    def test_at_20c_returns_original_values(self, default_kinetic_params):
        """At 20 degC the adjusted values should equal the 20 degC reference values."""
        result = adjust_temperature(default_kinetic_params, 20.0)
        np.testing.assert_allclose(result.MuHatHetero, result.MuHatHetero20)
        np.testing.assert_allclose(result.Ks, result.Ks20)
        np.testing.assert_allclose(result.Bh, result.Bh20)
        np.testing.assert_allclose(result.Kmp, result.Kmp20)
        np.testing.assert_allclose(result.Ksp, result.Ksp20)
        np.testing.assert_allclose(result.Ka, result.Ka20)
        np.testing.assert_allclose(result.Kr, result.Kr20)
        np.testing.assert_allclose(result.MuHatAuto, result.MuHatAuto20)
        np.testing.assert_allclose(result.Knh, result.Knh20)
        np.testing.assert_allclose(result.Ba, result.Ba20)

    def test_at_15c_produces_correct_adjustments(self, default_kinetic_params):
        """At 15 degC, values should be adjusted by theta^(-5)."""
        result = adjust_temperature(default_kinetic_params, 15.0)
        kp = default_kinetic_params
        td = 15.0 - 20.0  # = -5.0

        np.testing.assert_allclose(
            result.MuHatHetero,
            kp.MuHatHetero20 * kp.ThetaMuHatH ** td,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.Bh,
            kp.Bh20 * kp.ThetaBh ** td,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.MuHatAuto,
            kp.MuHatAuto20 * kp.ThetaMuHatA ** td,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.Ba,
            kp.Ba20 * kp.ThetaBa ** td,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.Knh,
            kp.Knh20 * kp.ThetaKnh ** td,
            rtol=1e-12,
        )

    def test_at_25c_produces_correct_adjustments(self, default_kinetic_params):
        """At 25 degC, values should be adjusted by theta^(+5)."""
        result = adjust_temperature(default_kinetic_params, 25.0)
        kp = default_kinetic_params
        td = 25.0 - 20.0  # = +5.0

        np.testing.assert_allclose(
            result.MuHatHetero,
            kp.MuHatHetero20 * kp.ThetaMuHatH ** td,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.Ks,
            kp.Ks20 * kp.ThetaKs ** td,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.Kmp,
            kp.Kmp20 * kp.ThetaKmp ** td,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.Ksp,
            kp.Ksp20 * kp.ThetaKsp ** td,
            rtol=1e-12,
        )

    def test_20c_reference_values_preserved(self, default_kinetic_params):
        """The 20 degC reference values should not change after adjustment."""
        result = adjust_temperature(default_kinetic_params, 15.0)
        kp = default_kinetic_params

        assert result.MuHatHetero20 == kp.MuHatHetero20
        assert result.Ks20 == kp.Ks20
        assert result.Bh20 == kp.Bh20
        assert result.Kmp20 == kp.Kmp20
        assert result.Ksp20 == kp.Ksp20
        assert result.Ka20 == kp.Ka20
        assert result.Kr20 == kp.Kr20
        assert result.MuHatAuto20 == kp.MuHatAuto20
        assert result.Knh20 == kp.Knh20
        assert result.Ba20 == kp.Ba20

    def test_returns_new_instance(self, default_kinetic_params):
        """adjust_temperature should return a new KineticParams instance."""
        result = adjust_temperature(default_kinetic_params, 15.0)
        assert result is not default_kinetic_params
        assert isinstance(result, KineticParams)

    def test_theta_coefficients_preserved(self, default_kinetic_params):
        """Theta (Arrhenius) coefficients should be unchanged."""
        result = adjust_temperature(default_kinetic_params, 30.0)
        assert result.ThetaMuHatH == default_kinetic_params.ThetaMuHatH
        assert result.ThetaBh == default_kinetic_params.ThetaBh
        assert result.ThetaMuHatA == default_kinetic_params.ThetaMuHatA

    def test_higher_temp_increases_mu_hat_hetero(self, default_kinetic_params):
        """MuHatHetero should increase with temperature (ThetaMuHatH > 1)."""
        result_low = adjust_temperature(default_kinetic_params, 15.0)
        result_high = adjust_temperature(default_kinetic_params, 25.0)
        assert result_high.MuHatHetero > result_low.MuHatHetero

    def test_ks_unchanged_with_theta_1(self, default_kinetic_params):
        """Ks has ThetaKs = 1.0, so it should be unchanged at any temperature."""
        result = adjust_temperature(default_kinetic_params, 10.0)
        np.testing.assert_allclose(result.Ks, default_kinetic_params.Ks20, rtol=1e-12)

    def test_ksp_decreases_with_higher_temp(self, default_kinetic_params):
        """Ksp has ThetaKsp = 0.91 < 1, so Ksp should decrease at higher temperatures."""
        result_20 = adjust_temperature(default_kinetic_params, 20.0)
        result_25 = adjust_temperature(default_kinetic_params, 25.0)
        assert result_25.Ksp < result_20.Ksp
