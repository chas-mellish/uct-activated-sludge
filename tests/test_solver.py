"""Tests for the Newton-Raphson solver with Gauss elimination."""

import numpy as np
import pytest

from uct_activated_sludge.constants import MAX_N, MAX_REAC_P1, TOTAL_COMPOUNDS
from uct_activated_sludge.models import PlantConfig, _zeros_2d_1based
from uct_activated_sludge.solver import gauss, newton, scale_values


class TestGauss:
    """Tests for the Gauss elimination solver."""

    def test_2x2_system(self):
        """Solve a simple 2x2 system: x=1, y=2."""
        # 2x + 1y = 4
        # 1x + 3y = 7
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)

        A[1, 1] = 2.0
        A[1, 2] = 1.0
        A[2, 1] = 1.0
        A[2, 2] = 3.0
        b[1] = 4.0
        b[2] = 7.0

        x, success = gauss(2, A, b)
        assert success is True
        np.testing.assert_allclose(x[1], 1.0, atol=1e-10)
        np.testing.assert_allclose(x[2], 2.0, atol=1e-10)

    def test_3x3_system(self):
        """Solve a 3x3 system with known solution x=1, y=2, z=3."""
        # 2x + 1y + 1z = 7
        # 1x + 3y + 2z = 13
        # 1x + 1y + 4z = 15
        A = np.zeros((4, 4), dtype=np.float64)
        b = np.zeros(4, dtype=np.float64)

        A[1, 1] = 2.0; A[1, 2] = 1.0; A[1, 3] = 1.0
        A[2, 1] = 1.0; A[2, 2] = 3.0; A[2, 3] = 2.0
        A[3, 1] = 1.0; A[3, 2] = 1.0; A[3, 3] = 4.0
        b[1] = 7.0
        b[2] = 13.0
        b[3] = 15.0

        x, success = gauss(3, A, b)
        assert success is True
        np.testing.assert_allclose(x[1], 1.0, atol=1e-10)
        np.testing.assert_allclose(x[2], 2.0, atol=1e-10)
        np.testing.assert_allclose(x[3], 3.0, atol=1e-10)

    def test_singular_matrix_detected(self):
        """Gauss should return success=False for a singular matrix."""
        # Rows 1 and 2 are identical
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)

        A[1, 1] = 1.0; A[1, 2] = 2.0
        A[2, 1] = 1.0; A[2, 2] = 2.0
        b[1] = 3.0
        b[2] = 3.0

        x, success = gauss(2, A, b)
        assert success is False

    def test_identity_matrix(self):
        """Identity matrix with arbitrary RHS should give x = b."""
        A = np.zeros((4, 4), dtype=np.float64)
        b = np.zeros(4, dtype=np.float64)

        A[1, 1] = 1.0; A[2, 2] = 1.0; A[3, 3] = 1.0
        b[1] = 5.0; b[2] = 10.0; b[3] = 15.0

        x, success = gauss(3, A, b)
        assert success is True
        np.testing.assert_allclose(x[1], 5.0, atol=1e-12)
        np.testing.assert_allclose(x[2], 10.0, atol=1e-12)
        np.testing.assert_allclose(x[3], 15.0, atol=1e-12)

    def test_diagonal_system(self):
        """Diagonal system A = diag(2,3,4), b = (6,9,12) => x = (3,3,3)."""
        A = np.zeros((4, 4), dtype=np.float64)
        b = np.zeros(4, dtype=np.float64)

        A[1, 1] = 2.0; A[2, 2] = 3.0; A[3, 3] = 4.0
        b[1] = 6.0; b[2] = 9.0; b[3] = 12.0

        x, success = gauss(3, A, b)
        assert success is True
        np.testing.assert_allclose(x[1], 3.0, atol=1e-12)
        np.testing.assert_allclose(x[2], 3.0, atol=1e-12)
        np.testing.assert_allclose(x[3], 3.0, atol=1e-12)

    def test_gauss_modifies_a_and_b_inplace(self):
        """Gauss should modify A and b in place (standard for elimination)."""
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        A[1, 1] = 2.0; A[1, 2] = 1.0
        A[2, 1] = 1.0; A[2, 2] = 3.0
        b[1] = 4.0; b[2] = 7.0

        A_copy = A.copy()
        b_copy = b.copy()

        gauss(2, A, b)
        # After Gauss, the arrays should be modified
        assert not np.array_equal(A, A_copy) or not np.array_equal(b, b_copy)


class TestScaleValues:
    """Tests for the scale_values function."""

    def test_returns_correct_shape(self, default_plant_config):
        """Scale vector should have shape (MAX_N + 1,)."""
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        scale = scale_values(C, default_plant_config, compounds=13)
        assert scale.shape == (MAX_N + 1,)

    def test_zero_concentrations_give_scale_one(self, default_plant_config):
        """When concentrations are zero, scale should be 1.0."""
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        scale = scale_values(C, default_plant_config, compounds=13)
        # For reactors 1..LastReactor+1 and compounds 1..13
        for k in range(1, default_plant_config.LastReactor + 2):
            for i in range(1, 14):
                idx = 13 * (k - 1) + i
                assert scale[idx] == 1.0

    def test_nonzero_concentrations_give_inverse(self, default_plant_config):
        """When concentration is positive, scale should be 1/C."""
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        C[1, 1] = 100.0
        C[1, 2] = 50.0
        C[2, 1] = 200.0

        scale = scale_values(C, default_plant_config, compounds=13)
        # C[1,1] = 100 => scale[idx] = 1/100 = 0.01
        np.testing.assert_allclose(scale[13 * 0 + 1], 0.01, rtol=1e-12)
        np.testing.assert_allclose(scale[13 * 0 + 2], 0.02, rtol=1e-12)
        np.testing.assert_allclose(scale[13 * 1 + 1], 0.005, rtol=1e-12)

    def test_produces_nonzero_scales(self, default_plant_config):
        """All scale values in the active range should be non-zero (either 1.0 or 1/C)."""
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        for k in range(1, default_plant_config.LastReactor + 2):
            for i in range(1, 14):
                C[k, i] = float(k * 13 + i)

        scale = scale_values(C, default_plant_config, compounds=13)
        for k in range(1, default_plant_config.LastReactor + 2):
            for i in range(1, 14):
                idx = 13 * (k - 1) + i
                assert scale[idx] != 0.0


class TestNewton:
    """Tests for the newton() solver directly."""

    def test_tracer_mode_converges(self):
        """Newton in tracer mode (compounds=1) should converge on a simple config."""
        from uct_activated_sludge.models import (
            KineticParams,
            StoichiometricParams,
        )
        from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix
        from uct_activated_sludge.kinetics import air_supply
        from uct_activated_sludge.hydraulics import flow_division_ss_inplace

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
        pc.FlowWaste = pc.VolumeTotal / pc.Rs

        # Set up steady-state flows
        flow_division_ss_inplace(pc)

        kp = KineticParams()
        sp = StoichiometricParams()
        stoich = build_stoichiometric_matrix(sp)

        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, kp)

        # Seed tracer concentrations (compound 1)
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        tracer_conc_in = 100.0
        for k in range(1, pc.LastReactor + 1):
            C[k, 1] = pc.FlowFeed * tracer_conc_in * pc.Rs / pc.VolumeTotal
        C[pc.LastReactor + 1, 1] = (
            (pc.FlowFeed + pc.FlowRASrecycle - pc.FlowWaste)
            / pc.FlowRASrecycle
            * C[pc.LastReactor, 1]
        )

        C0 = np.zeros(TOTAL_COMPOUNDS + 1, dtype=np.float64)

        C_result, converged = newton(
            C, 1, pc, kp, sp, stoich, C0,
            air_on_h, air_off_h, air_on_a, air_off_a,
            TracerConcIn=tracer_conc_in,
        )

        assert converged is True

        # Tracer concentrations should be positive and finite
        for k in range(1, pc.LastReactor + 2):
            assert np.isfinite(C_result[k, 1])
            assert C_result[k, 1] > 0.0
