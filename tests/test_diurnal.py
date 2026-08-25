"""Tier 2 integration tests for the diurnal dynamic simulation module."""

import numpy as np
import pytest

from uct_activated_sludge.diurnal import (
    DiurnalData,
    default_diurnal_data,
    run_diurnal,
)


class TestDefaultDiurnalData:
    """Tests for default_diurnal_data."""

    def test_returns_12_records(self):
        """Should return exactly 12 records for 2-hourly intervals."""
        data = default_diurnal_data()
        assert len(data) == 12

    def test_record_types(self):
        """All records should be DiurnalData instances."""
        data = default_diurnal_data()
        for rec in data:
            assert isinstance(rec, DiurnalData)

    def test_times_are_0_to_22(self):
        """Times should be 0, 2, 4, ..., 22 hours."""
        data = default_diurnal_data()
        expected_times = [float(i * 2) for i in range(12)]
        actual_times = [rec.Time for rec in data]
        assert actual_times == expected_times

    def test_default_values(self):
        """Default flow, COD, TKN should match the function defaults."""
        data = default_diurnal_data()
        for rec in data:
            assert rec.Flow == 15.0
            assert rec.COD == 500.0
            assert rec.TKN == 50.0

    def test_custom_values(self):
        """Custom flow, COD, TKN should be propagated to all records."""
        data = default_diurnal_data(flow=30.0, cod=800.0, tkn=80.0)
        for rec in data:
            assert rec.Flow == 30.0
            assert rec.COD == 800.0
            assert rec.TKN == 80.0


class TestDiurnalData:
    """Tests for the DiurnalData dataclass."""

    def test_default_construction(self):
        """Default DiurnalData should have all zeros."""
        d = DiurnalData()
        assert d.Time == 0.0
        assert d.Flow == 0.0
        assert d.COD == 0.0
        assert d.TKN == 0.0

    def test_custom_construction(self):
        """DiurnalData should accept custom values."""
        d = DiurnalData(Time=6.0, Flow=20.0, COD=600.0, TKN=60.0)
        assert d.Time == 6.0
        assert d.Flow == 20.0
        assert d.COD == 600.0
        assert d.TKN == 60.0


class TestRunDiurnalImport:
    """Basic import tests for the diurnal module."""

    def test_run_diurnal_is_callable(self):
        """run_diurnal should be importable and callable."""
        assert callable(run_diurnal)

    def test_store_response_importable(self):
        """store_response should be importable."""
        from uct_activated_sludge.diurnal import store_response
        assert callable(store_response)

    def test_modify_waste_importable(self):
        """modify_waste should be importable."""
        from uct_activated_sludge.diurnal import modify_waste
        assert callable(modify_waste)


class TestRunDiurnalExecution:
    """Integration tests that actually call run_diurnal with a real config."""

    @pytest.fixture
    def steady_state_result(self):
        """Run a steady-state solve to get initial conditions for diurnal."""
        from uct_activated_sludge.steady_state import run_steady_state
        from uct_activated_sludge.models import (
            KineticParams,
            PlantConfig,
            StoichiometricParams,
            WastewaterParams,
        )

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

        return run_steady_state(pc, KineticParams(), StoichiometricParams(), WastewaterParams())

    def test_run_diurnal_with_default_data(self, steady_state_result):
        """run_diurnal should complete and return expected keys."""
        from uct_activated_sludge.models import (
            IntegrationParams,
            KineticParams,
            StoichiometricParams,
            WastewaterParams,
        )

        ss = steady_state_result
        diurnal_data = default_diurnal_data(flow=25.0, cod=500.0, tkn=50.0)

        result = run_diurnal(
            plant_config=ss["plant_config"],
            kinetic_params=ss["kinetic_params"],
            stoich_params=StoichiometricParams(),
            ww_params=WastewaterParams(),
            integration_params=IntegrationParams(),
            C_steady=ss["CSteady"],
            diurnal_data=diurnal_data,
            stoich_matrix=ss["Stoich"],
            max_cycles=5,
        )

        # Verify expected keys
        expected_keys = {
            "response", "cycle_count", "converged",
            "C_final", "data_per_day", "data_int_hours",
        }
        assert expected_keys == set(result.keys())

    def test_run_diurnal_cycle_count_positive(self, steady_state_result):
        """run_diurnal should execute at least one cycle."""
        from uct_activated_sludge.models import (
            IntegrationParams,
            StoichiometricParams,
            WastewaterParams,
        )

        ss = steady_state_result
        diurnal_data = default_diurnal_data(flow=25.0, cod=500.0, tkn=50.0)

        result = run_diurnal(
            plant_config=ss["plant_config"],
            kinetic_params=ss["kinetic_params"],
            stoich_params=StoichiometricParams(),
            ww_params=WastewaterParams(),
            integration_params=IntegrationParams(),
            C_steady=ss["CSteady"],
            diurnal_data=diurnal_data,
            stoich_matrix=ss["Stoich"],
            max_cycles=5,
        )

        assert result["cycle_count"] >= 1

    def test_run_diurnal_concentrations_non_negative(self, steady_state_result):
        """Final concentrations should be finite and non-negative."""
        from uct_activated_sludge.models import (
            IntegrationParams,
            StoichiometricParams,
            WastewaterParams,
        )

        ss = steady_state_result
        diurnal_data = default_diurnal_data(flow=25.0, cod=500.0, tkn=50.0)

        result = run_diurnal(
            plant_config=ss["plant_config"],
            kinetic_params=ss["kinetic_params"],
            stoich_params=StoichiometricParams(),
            ww_params=WastewaterParams(),
            integration_params=IntegrationParams(),
            C_steady=ss["CSteady"],
            diurnal_data=diurnal_data,
            stoich_matrix=ss["Stoich"],
            max_cycles=3,
        )

        C_final = result["C_final"]
        last_reactor = ss["plant_config"].LastReactor
        for k in range(1, last_reactor + 1):
            for i in range(1, 14):
                assert np.isfinite(C_final[k, i]), (
                    f"C_final[{k},{i}] is not finite: {C_final[k, i]}"
                )
                assert C_final[k, i] >= 0.0, (
                    f"C_final[{k},{i}] is negative: {C_final[k, i]}"
                )
