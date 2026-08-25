"""Tests for the ASM1 kinetic rate expressions, air supply, and utilization rates."""

import numpy as np
import pytest

from uct_activated_sludge.constants import (
    MAX_REAC_P1,
    NO_PROCESSES,
    TOTAL_COMPOUNDS,
)
from uct_activated_sludge.kinetics import air_supply, process_rates, utilization_rates
from uct_activated_sludge.models import (
    KineticParams,
    PlantConfig,
    StoichiometricParams,
    _zeros_2d_1based,
)
from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix


class TestProcessRates:
    """Tests for the process_rates function."""

    def test_returns_correct_shape(self, default_kinetic_params, default_stoich_params, default_plant_config):
        """process_rates should return an array of shape (NO_PROCESSES+1,)."""
        pc = default_plant_config
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, default_kinetic_params)

        # Set up a concentration matrix with some positive values
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        for k in range(1, pc.LastReactor + 1):
            for i in range(1, TOTAL_COMPOUNDS + 1):
                C[k, i] = 1.0

        Rho = process_rates(
            C, 1, default_kinetic_params, default_stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        assert Rho.shape == (NO_PROCESSES + 1,)

    def test_zero_concentrations_give_zero_rates(self, default_kinetic_params, default_stoich_params, default_plant_config):
        """With all-zero concentrations, all process rates should be zero."""
        pc = default_plant_config
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, default_kinetic_params)
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)

        Rho = process_rates(
            C, 1, default_kinetic_params, default_stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        np.testing.assert_array_equal(Rho[1:], 0.0)

    def test_positive_concentrations_produce_nonnegative_rates(
        self, default_kinetic_params, default_stoich_params, default_plant_config
    ):
        """With realistic positive concentrations, all process rates should be >= 0."""
        pc = default_plant_config
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, default_kinetic_params)

        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        # Set typical reactor 3 (aerobic) concentrations
        k = 3
        C[k, 1] = 100.0   # Xbh
        C[k, 2] = 20.0    # Xe
        C[k, 3] = 5.0     # Xba
        C[k, 4] = 10.0    # Xs
        C[k, 5] = 50.0    # Sbp
        C[k, 6] = 30.0    # Xi
        C[k, 7] = 5.0     # Xnd
        C[k, 8] = 5.0     # Ss
        C[k, 9] = 2.0     # Snh
        C[k, 10] = 1.0    # Snd
        C[k, 11] = 3.0    # Sno
        C[k, 12] = 5.0    # Alk
        C[k, 13] = 25.0   # Si

        Rho = process_rates(
            C, k, default_kinetic_params, default_stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        # All rates should be non-negative
        for j in range(1, NO_PROCESSES + 1):
            assert Rho[j] >= 0.0, f"Rho[{j}] = {Rho[j]} is negative"

    def test_decay_rate_proportional_to_biomass(self, default_kinetic_params, default_stoich_params, default_plant_config):
        """Decay rate (process 9) should be Bh * Xbh."""
        pc = default_plant_config
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, default_kinetic_params)

        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        C[1, 1] = 200.0  # Xbh in reactor 1

        Rho = process_rates(
            C, 1, default_kinetic_params, default_stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        expected = default_kinetic_params.Bh * 200.0
        np.testing.assert_allclose(Rho[9], expected, rtol=1e-12)


class TestAirSupply:
    """Tests for the air_supply function."""

    def test_returns_four_arrays(self, default_plant_config, default_kinetic_params):
        """air_supply should return a tuple of 4 arrays."""
        result = air_supply(default_plant_config, default_kinetic_params)
        assert len(result) == 4
        for arr in result:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (MAX_REAC_P1 + 1,)

    def test_unaerated_reactor_switching(self, default_plant_config, default_kinetic_params):
        """In unaerated reactors (DOConc=0), air_on should be 0 and air_off should be 1."""
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(
            default_plant_config, default_kinetic_params
        )
        # Reactor 1 has DOConc = 0 (anaerobic)
        np.testing.assert_allclose(air_on_h[1], 0.0, atol=1e-15)
        np.testing.assert_allclose(air_off_h[1], 1.0, atol=1e-15)
        np.testing.assert_allclose(air_on_a[1], 0.0, atol=1e-15)
        np.testing.assert_allclose(air_off_a[1], 1.0, atol=1e-15)

    def test_aerated_reactor_switching(self, default_plant_config, default_kinetic_params):
        """In aerated reactor (DOConc=2.0), air_on should be close to 1."""
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(
            default_plant_config, default_kinetic_params
        )
        # Reactor 3 has DOConc = 2.0, Koh = 0.002
        # air_on = 2.0 / (0.002 + 2.0) ~ 0.999
        assert air_on_h[3] > 0.99
        assert air_off_h[3] < 0.01

    def test_on_plus_off_equals_one(self, default_plant_config, default_kinetic_params):
        """air_on + air_off should equal 1.0 for each reactor (hetero and auto)."""
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(
            default_plant_config, default_kinetic_params
        )
        for k in range(1, default_plant_config.LastReactor + 1):
            np.testing.assert_allclose(
                air_on_h[k] + air_off_h[k], 1.0, atol=1e-14,
                err_msg=f"Hetero on+off != 1 at reactor {k}",
            )
            np.testing.assert_allclose(
                air_on_a[k] + air_off_a[k], 1.0, atol=1e-14,
                err_msg=f"Auto on+off != 1 at reactor {k}",
            )


class TestUtilizationRates:
    """Tests for the utilization_rates function."""

    def test_returns_four_arrays(self, default_plant_config, default_kinetic_params, default_stoich_params, default_stoich_matrix):
        """utilization_rates should return (Oc, On, Ot, Denit)."""
        pc = default_plant_config
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, default_kinetic_params)
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)

        Oc, On, Ot, Denit = utilization_rates(
            C, default_stoich_matrix, pc, default_kinetic_params,
            default_stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        assert Oc.shape == (MAX_REAC_P1 + 1,)
        assert On.shape == (MAX_REAC_P1 + 1,)
        assert Ot.shape == (MAX_REAC_P1 + 1,)
        assert Denit.shape == (MAX_REAC_P1 + 1,)

    def test_ot_equals_oc_plus_on(self, default_plant_config, default_kinetic_params, default_stoich_params, default_stoich_matrix):
        """Total OUR (Ot) should equal Oc + On for each reactor."""
        pc = default_plant_config
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, default_kinetic_params)

        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)
        # Set some concentrations in reactor 3
        C[3, 1] = 100.0
        C[3, 3] = 5.0
        C[3, 4] = 10.0
        C[3, 5] = 50.0
        C[3, 8] = 5.0
        C[3, 9] = 2.0
        C[3, 11] = 3.0

        Oc, On, Ot, Denit = utilization_rates(
            C, default_stoich_matrix, pc, default_kinetic_params,
            default_stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        for k in range(1, pc.LastReactor + 1):
            np.testing.assert_allclose(
                Ot[k], Oc[k] + On[k], atol=1e-12,
                err_msg=f"Ot != Oc + On at reactor {k}",
            )

    def test_zero_concentrations_give_zero_rates(self, default_plant_config, default_kinetic_params, default_stoich_params, default_stoich_matrix):
        """With zero concentrations, all utilization rates should be zero."""
        pc = default_plant_config
        air_on_h, air_off_h, air_on_a, air_off_a = air_supply(pc, default_kinetic_params)
        C = _zeros_2d_1based(MAX_REAC_P1, TOTAL_COMPOUNDS)

        Oc, On, Ot, Denit = utilization_rates(
            C, default_stoich_matrix, pc, default_kinetic_params,
            default_stoich_params,
            air_on_h, air_off_h, air_on_a, air_off_a,
        )
        for k in range(1, pc.LastReactor + 1):
            np.testing.assert_allclose(Oc[k], 0.0, atol=1e-15)
            np.testing.assert_allclose(On[k], 0.0, atol=1e-15)
            np.testing.assert_allclose(Denit[k], 0.0, atol=1e-15)
