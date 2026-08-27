"""Functional tests for the uct-as CLI tool.

Tests the CLI entry point (uct-as) for:
- Help output
- Version output
- Parameter display (kinetics, stoichiometry, wastewater, plant, integration)
- Steady-state simulation with a valid config
- Error handling (no config, bad config path, unconfigured plant)
"""

import os
import subprocess
import json
import tempfile

import pytest

WORKSPACE_DIR = os.environ.get(
    "WORKSPACE_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)
REPO_DIR = os.path.join(WORKSPACE_DIR, "uct-activated-sludge")
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CONFIG = os.path.join(TEST_DIR, "test_plant.toml")
EXAMPLES_CONFIG = os.path.join(REPO_DIR, "examples", "default_plant.toml")


def run_cli(*args, input_text=None, timeout=60):
    """Helper to invoke the uct-as CLI and capture output."""
    result = subprocess.run(
        ["uct-as", *args],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )
    return result


class TestHelpOutput:
    """uct-as --help -- verify help text and exit code."""

    def test_help_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0

    def test_help_shows_description(self):
        result = run_cli("--help")
        assert "ASM1" in result.stdout or "Activated Sludge" in result.stdout

    def test_help_shows_subcommands(self):
        result = run_cli("--help")
        assert "steady-state" in result.stdout
        assert "diurnal" in result.stdout
        assert "params" in result.stdout

    def test_no_args_shows_help(self):
        """Running uct-as with no args should show help and exit 0."""
        result = run_cli()
        assert result.returncode == 0
        assert "steady-state" in result.stdout


class TestVersionOutput:
    """uct-as --version -- verify version string."""

    def test_version_exits_zero(self):
        result = run_cli("--version")
        assert result.returncode == 0

    def test_version_shows_semver(self):
        result = run_cli("--version")
        assert "0.1.0" in result.stdout


class TestParamsKinetics:
    """uct-as params --list kinetics -- display kinetic parameter defaults."""

    def test_params_kinetics_exits_zero(self):
        result = run_cli("params", "--list", "kinetics")
        assert result.returncode == 0

    def test_params_kinetics_shows_header(self):
        result = run_cli("params", "--list", "kinetics")
        assert "KineticParams" in result.stdout

    def test_params_kinetics_shows_key_params(self):
        result = run_cli("params", "--list", "kinetics")
        assert "MuHatHetero20" in result.stdout
        assert "3.2" in result.stdout
        assert "Ks20" in result.stdout
        assert "Bh20" in result.stdout
        assert "MuHatAuto20" in result.stdout


class TestParamsStoichiometry:
    """uct-as params --list stoichiometry -- display stoichiometric defaults."""

    def test_params_stoichiometry_exits_zero(self):
        result = run_cli("params", "--list", "stoichiometry")
        assert result.returncode == 0

    def test_params_stoichiometry_shows_header(self):
        result = run_cli("params", "--list", "stoichiometry")
        assert "StoichiometricParams" in result.stdout

    def test_params_stoichiometry_shows_key_params(self):
        result = run_cli("params", "--list", "stoichiometry")
        assert "Yh" in result.stdout
        assert "0.666" in result.stdout
        assert "Ya" in result.stdout
        assert "CODVSS" in result.stdout


class TestParamsWastewater:
    """uct-as params --list wastewater -- display wastewater defaults."""

    def test_params_wastewater_exits_zero(self):
        result = run_cli("params", "--list", "wastewater")
        assert result.returncode == 0

    def test_params_wastewater_shows_key_params(self):
        result = run_cli("params", "--list", "wastewater")
        assert "Sti" in result.stdout
        assert "500" in result.stdout
        assert "Nti" in result.stdout


class TestParamsPlant:
    """uct-as params --list plant -- display plant config defaults."""

    def test_params_plant_exits_zero(self):
        result = run_cli("params", "--list", "plant")
        assert result.returncode == 0

    def test_params_plant_shows_key_fields(self):
        result = run_cli("params", "--list", "plant")
        assert "PlantConfig" in result.stdout
        assert "LastReactor" in result.stdout
        assert "FlowFeed" in result.stdout


class TestParamsIntegration:
    """uct-as params --list integration -- display integration defaults."""

    def test_params_integration_exits_zero(self):
        result = run_cli("params", "--list", "integration")
        assert result.returncode == 0

    def test_params_integration_shows_key_fields(self):
        result = run_cli("params", "--list", "integration")
        assert "IntegrationParams" in result.stdout
        assert "Accuracy" in result.stdout


class TestParamsNoGroup:
    """uct-as params without --list -- should return error."""

    def test_params_no_list_exits_nonzero(self):
        result = run_cli("params")
        assert result.returncode != 0


class TestParamsInvalidGroup:
    """uct-as params --list invalid -- should return error."""

    def test_params_invalid_group_exits_nonzero(self):
        result = run_cli("params", "--list", "nonexistent")
        assert result.returncode != 0


class TestSteadyStateNoConfig:
    """uct-as steady-state with no config -- error handling for unconfigured plant."""

    def test_steady_state_no_config_exits_nonzero(self):
        """With default params (FlowFeed=0, Rs=0), should fail with validation error."""
        result = run_cli("steady-state")
        assert result.returncode != 0

    def test_steady_state_no_config_error_message(self):
        result = run_cli("steady-state")
        # Should report that Rs or FlowFeed must be > 0
        assert "Error" in result.stderr or "error" in result.stderr.lower()


class TestSteadyStateBadConfigPath:
    """uct-as steady-state --config nonexistent.toml -- error handling."""

    def test_bad_config_path_exits_nonzero(self):
        result = run_cli("steady-state", "--config", "/nonexistent/path/foo.toml")
        assert result.returncode != 0

    def test_bad_config_path_error_message(self):
        result = run_cli("steady-state", "--config", "/nonexistent/path/foo.toml")
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestSteadyStateDefaultPlantToml:
    """uct-as steady-state --config examples/default_plant.toml -- error because Rs=0."""

    def test_default_plant_config_exits_nonzero(self):
        """The shipped default_plant.toml has Rs=0.0, FlowFeed=0.0 -- validation should catch this."""
        result = run_cli("steady-state", "--config", EXAMPLES_CONFIG)
        assert result.returncode != 0

    def test_default_plant_config_error_mentions_validation(self):
        result = run_cli("steady-state", "--config", EXAMPLES_CONFIG)
        assert "Error" in result.stderr


class TestSteadyStateValidConfig:
    """uct-as steady-state --config test_plant.toml -- full simulation run."""

    def test_steady_state_exits_zero(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_steady_state_shows_results(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
        assert "STEADY STATE RESULTS" in result.stdout

    def test_steady_state_shows_compound_names(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
        # Check for some ASM1 compound names in the output
        stdout = result.stdout
        assert "Xbh" in stdout or "Heterotrophic" in stdout or "biomass" in stdout.lower()

    def test_steady_state_shows_vss(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
        assert "Volatile SS" in result.stdout or "VSS" in result.stdout

    def test_steady_state_shows_our(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
        assert "OUR" in result.stdout

    def test_steady_state_json_output(self):
        """Test that --output flag produces valid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        try:
            result = run_cli(
                "steady-state", "--config", TEST_CONFIG,
                "--output", output_path,
                timeout=120,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            with open(output_path) as fh:
                data = json.load(fh)
            # Check that the JSON has the expected keys
            assert "C" in data
            assert "C0" in data
            assert "converged" in data
            assert "FlowWaste" in data
            assert "Oc" in data
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_steady_state_json_concentrations_positive(self):
        """Check that simulation produces positive concentrations."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        try:
            result = run_cli(
                "steady-state", "--config", TEST_CONFIG,
                "--output", output_path,
                timeout=120,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            with open(output_path) as fh:
                data = json.load(fh)
            # C is a 2D array; check that reactor 1 (index 1) concentrations
            # are non-negative (compound indices 1..13)
            C = data["C"]
            for k in range(1, 4):  # reactors 1-3
                for i in range(1, 14):  # compounds 1-13
                    assert C[k][i] >= 0.0, (
                        f"Negative concentration at reactor {k}, compound {i}: {C[k][i]}"
                    )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_steady_state_converged(self):
        """Check that the Newton solver converged."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        try:
            result = run_cli(
                "steady-state", "--config", TEST_CONFIG,
                "--output", output_path,
                timeout=120,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            with open(output_path) as fh:
                data = json.load(fh)
            assert data["converged"] is True, "Newton solver did not converge"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestSteadyStateSubcommandHelp:
    """uct-as steady-state --help -- check subcommand help."""

    def test_steady_state_help_exits_zero(self):
        result = run_cli("steady-state", "--help")
        assert result.returncode == 0

    def test_steady_state_help_mentions_config(self):
        result = run_cli("steady-state", "--help")
        assert "--config" in result.stdout


class TestDiurnalSubcommandHelp:
    """uct-as diurnal --help -- check subcommand help."""

    def test_diurnal_help_exits_zero(self):
        result = run_cli("diurnal", "--help")
        assert result.returncode == 0

    def test_diurnal_help_mentions_config(self):
        result = run_cli("diurnal", "--help")
        assert "--config" in result.stdout
        assert "--diurnal-data" in result.stdout
