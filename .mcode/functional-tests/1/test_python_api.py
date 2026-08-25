"""Functional tests for the uct-activated-sludge Python API.

Tests the importability and correctness of public APIs:
- Package import and version
- Stoichiometric matrix construction
- Temperature adjustment
- Influent fractionation
- Steady-state simulation via Python API
"""

import numpy as np
import pytest


class TestPackageImport:
    """Verify the package is importable and exports the expected version."""

    def test_import_version(self):
        from uct_activated_sludge import __version__
        assert __version__ == "0.1.0"

    def test_import_models(self):
        from uct_activated_sludge.models import (
            KineticParams,
            StoichiometricParams,
            WastewaterParams,
            PlantConfig,
            ReactorState,
            IntegrationParams,
        )
        # Verify they are instantiable with defaults
        kp = KineticParams()
        assert kp.MuHatHetero20 == 3.2

        sp = StoichiometricParams()
        assert sp.Yh == pytest.approx(0.666, rel=1e-3)

        wp = WastewaterParams()
        assert wp.Sti == 500.0

    def test_import_stoichiometry(self):
        from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix
        assert callable(build_stoichiometric_matrix)

    def test_import_temperature(self):
        from uct_activated_sludge.temperature import adjust_temperature
        assert callable(adjust_temperature)

    def test_import_kinetics(self):
        from uct_activated_sludge.kinetics import (
            process_rates,
            air_supply,
            utilization_rates,
        )
        assert callable(process_rates)

    def test_import_influent(self):
        from uct_activated_sludge.influent import fractionate_influent
        assert callable(fractionate_influent)

    def test_import_solver(self):
        from uct_activated_sludge.solver import newton, gauss
        assert callable(newton)
        assert callable(gauss)

    def test_import_steady_state(self):
        from uct_activated_sludge.steady_state import run_steady_state
        assert callable(run_steady_state)

    def test_import_config(self):
        from uct_activated_sludge.config import load_config, build_params_from_config
        assert callable(load_config)
        assert callable(build_params_from_config)


class TestStoichiometricMatrix:
    """Verify stoichiometric matrix has correct structure and key entries."""

    def test_matrix_shape(self):
        from uct_activated_sludge.models import StoichiometricParams
        from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix

        sp = StoichiometricParams()
        S = build_stoichiometric_matrix(sp)
        # 15x15 matrix (0-padded, 1-based: indices 1..14)
        assert S.shape == (15, 15)

    def test_aerobic_growth_ss_consumption(self):
        """Stoich[8,1] should equal -1/Yh (Ss consumption for aerobic growth)."""
        from uct_activated_sludge.models import StoichiometricParams
        from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix

        sp = StoichiometricParams()
        S = build_stoichiometric_matrix(sp)
        expected = -1.0 / sp.Yh
        assert S[8, 1] == pytest.approx(expected, rel=1e-6)

    def test_matrix_not_all_zeros(self):
        from uct_activated_sludge.models import StoichiometricParams
        from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix

        sp = StoichiometricParams()
        S = build_stoichiometric_matrix(sp)
        # Should have many non-zero entries
        assert np.count_nonzero(S) > 20


class TestTemperatureAdjustment:
    """Verify Arrhenius temperature corrections."""

    def test_at_20_degrees_no_change(self):
        from uct_activated_sludge.models import KineticParams
        from uct_activated_sludge.temperature import adjust_temperature

        kp = KineticParams()
        adjusted = adjust_temperature(kp, 20.0)
        assert adjusted.MuHatHetero == pytest.approx(kp.MuHatHetero20, rel=1e-6)
        assert adjusted.Ks == pytest.approx(kp.Ks20, rel=1e-6)

    def test_at_15_degrees_decreases(self):
        from uct_activated_sludge.models import KineticParams
        from uct_activated_sludge.temperature import adjust_temperature

        kp = KineticParams()
        adjusted = adjust_temperature(kp, 15.0)
        # MuHatHetero should decrease at lower temperature (ThetaMuHatH > 1)
        assert adjusted.MuHatHetero < kp.MuHatHetero20

    def test_at_25_degrees_increases(self):
        from uct_activated_sludge.models import KineticParams
        from uct_activated_sludge.temperature import adjust_temperature

        kp = KineticParams()
        adjusted = adjust_temperature(kp, 25.0)
        # MuHatHetero should increase at higher temperature
        assert adjusted.MuHatHetero > kp.MuHatHetero20


class TestInfluentFractionation:
    """Verify influent fractionation produces valid concentration vectors."""

    def test_fractionation_positive_concentrations(self):
        from uct_activated_sludge.models import StoichiometricParams, WastewaterParams
        from uct_activated_sludge.influent import fractionate_influent

        wp = WastewaterParams()
        sp = StoichiometricParams()
        C0 = fractionate_influent(wp, sp)

        # Check all 13 compounds are non-negative
        for i in range(1, 14):
            assert C0[i] >= 0.0, f"Negative C0[{i}] = {C0[i]}"

    def test_fractionation_cod_balance(self):
        """Total influent COD should be approximately Sti."""
        from uct_activated_sludge.models import StoichiometricParams, WastewaterParams
        from uct_activated_sludge.influent import fractionate_influent

        wp = WastewaterParams()
        sp = StoichiometricParams()
        C0 = fractionate_influent(wp, sp)

        # COD components: Xbh(1), Xs(4), Sbp(5), Xi(6), Ss(8), Si(13)
        # The sum should approximate Sti
        cod_sum = C0[1] + C0[4] + C0[5] + C0[6] + C0[8] + C0[13]
        # Allow some tolerance since fractionation redistributes
        assert cod_sum > 0.0
        assert cod_sum == pytest.approx(wp.Sti, rel=0.1)


class TestSteadyStatePythonAPI:
    """Test the run_steady_state Python API directly."""

    @pytest.fixture
    def configured_plant(self):
        """Build a properly configured 3-reactor UCT plant."""
        from uct_activated_sludge.models import (
            KineticParams,
            PlantConfig,
            StoichiometricParams,
            WastewaterParams,
        )

        kp = KineticParams()
        sp = StoichiometricParams()
        wp = WastewaterParams()
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

        return kp, sp, wp, pc

    def test_run_steady_state_returns_dict(self, configured_plant):
        from uct_activated_sludge.steady_state import run_steady_state

        kp, sp, wp, pc = configured_plant
        result = run_steady_state(pc, kp, sp, wp)
        assert isinstance(result, dict)

    def test_run_steady_state_has_expected_keys(self, configured_plant):
        from uct_activated_sludge.steady_state import run_steady_state

        kp, sp, wp, pc = configured_plant
        result = run_steady_state(pc, kp, sp, wp)
        expected_keys = {"C", "C0", "CSteady", "Stoich", "Oc", "On", "Ot",
                         "Denit", "converged", "output", "FlowWaste"}
        assert expected_keys.issubset(set(result.keys()))

    def test_run_steady_state_converges(self, configured_plant):
        from uct_activated_sludge.steady_state import run_steady_state

        kp, sp, wp, pc = configured_plant
        result = run_steady_state(pc, kp, sp, wp)
        assert result["converged"] is True

    def test_run_steady_state_positive_concentrations(self, configured_plant):
        from uct_activated_sludge.steady_state import run_steady_state

        kp, sp, wp, pc = configured_plant
        result = run_steady_state(pc, kp, sp, wp)
        C = result["C"]
        for k in range(1, 4):
            for i in range(1, 14):
                assert C[k, i] >= 0.0, (
                    f"Negative concentration at reactor {k}, compound {i}: {C[k, i]}"
                )

    def test_run_steady_state_positive_flowwaste(self, configured_plant):
        from uct_activated_sludge.steady_state import run_steady_state

        kp, sp, wp, pc = configured_plant
        result = run_steady_state(pc, kp, sp, wp)
        assert result["FlowWaste"] > 0.0

    def test_run_steady_state_output_text(self, configured_plant):
        from uct_activated_sludge.steady_state import run_steady_state

        kp, sp, wp, pc = configured_plant
        result = run_steady_state(pc, kp, sp, wp)
        assert "STEADY STATE RESULTS" in result["output"]

    def test_run_steady_state_validation_error_rs_zero(self):
        """Rs=0 should raise ValueError."""
        from uct_activated_sludge.models import (
            KineticParams, PlantConfig, StoichiometricParams, WastewaterParams,
        )
        from uct_activated_sludge.steady_state import run_steady_state

        pc = PlantConfig()
        pc.Rs = 0.0
        pc.FlowFeed = 25.0
        with pytest.raises(ValueError, match="Rs"):
            run_steady_state(pc, KineticParams(), StoichiometricParams(), WastewaterParams())

    def test_run_steady_state_validation_error_flowfeed_zero(self):
        """FlowFeed=0 should raise ValueError."""
        from uct_activated_sludge.models import (
            KineticParams, PlantConfig, StoichiometricParams, WastewaterParams,
        )
        from uct_activated_sludge.steady_state import run_steady_state

        pc = PlantConfig()
        pc.Rs = 20.0
        pc.FlowFeed = 0.0
        with pytest.raises(ValueError, match="FlowFeed"):
            run_steady_state(pc, KineticParams(), StoichiometricParams(), WastewaterParams())


class TestConfigLoading:
    """Verify TOML config loading and parameter construction."""

    def test_load_valid_config(self):
        import os
        from uct_activated_sludge.config import load_config

        config = load_config(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_plant.toml",
        ))
        assert "plant" in config
        assert "kinetics" in config
        assert config["plant"]["FlowFeed"] == 25.0

    def test_load_nonexistent_config_raises(self):
        from uct_activated_sludge.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.toml")

    def test_build_params_from_config(self):
        import os
        from uct_activated_sludge.config import load_config, build_params_from_config

        config = load_config(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_plant.toml",
        ))
        kp, sp, wp, pc, ip = build_params_from_config(config)
        assert pc.FlowFeed == 25.0
        assert pc.Rs == 20.0
        assert pc.LastReactor == 3
        assert sp.Yh == pytest.approx(0.666, rel=1e-3)
        assert wp.Sti == 500.0

    def test_build_params_empty_config_uses_defaults(self):
        from uct_activated_sludge.config import build_params_from_config

        kp, sp, wp, pc, ip = build_params_from_config({})
        assert kp.MuHatHetero20 == 3.2
        assert sp.Yh == pytest.approx(0.666, rel=1e-3)
        assert wp.Sti == 500.0
