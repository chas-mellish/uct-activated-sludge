"""Functional tests for the uct-asp CLI tool.

Tests the CLI entry point (uct-asp) for:
- Help output
- Version output
- Parameter display (kinetics, stoichiometry, wastewater, plant, integration)
- Steady-state simulation with a valid config
- Steady-state simulation with the shipped default_plant.toml
- Error handling (no config, bad config path, invalid param group)
- Diurnal subcommand help
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

# Path to the virtual environment's activate script
VENV_ACTIVATE = os.path.join(REPO_DIR, ".venv", "bin", "activate")


def run_cli(*args, input_text=None, timeout=60):
    """Helper to invoke the uct-asp CLI and capture output.

    Activates the virtual environment before running the command to ensure
    the uct-asp entry point is on PATH.
    """
    cmd = f". {VENV_ACTIVATE} && uct-asp {' '.join(args)}"
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )
    return result


class TestHelpOutput:
    """uct-asp --help -- verify help text and exit code."""

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

    def test_help_shows_prog_name(self):
        """Verify the prog name is uct-asp (not uct-as)."""
        result = run_cli("--help")
        assert "uct-asp" in result.stdout

    def test_no_args_shows_help(self):
        """Running uct-asp with no args should show help and exit 0."""
        result = run_cli()
        assert result.returncode == 0
        assert "steady-state" in result.stdout


class TestVersionOutput:
    """uct-asp --version -- verify version string."""

    def test_version_exits_zero(self):
        result = run_cli("--version")
        assert result.returncode == 0

    def test_version_shows_semver(self):
        result = run_cli("--version")
        assert "0.1.0" in result.stdout

    def test_version_shows_prog_name(self):
        """Verify the version output includes the correct prog name."""
        result = run_cli("--version")
        assert "uct-asp" in result.stdout


class TestParamsKinetics:
    """uct-asp params --group kinetics -- display kinetic parameter defaults."""

    def test_params_kinetics_exits_zero(self):
        result = run_cli("params", "--group", "kinetics")
        assert result.returncode == 0

    def test_params_kinetics_shows_header(self):
        result = run_cli("params", "--group", "kinetics")
        assert "KineticParams" in result.stdout

    def test_params_kinetics_shows_key_params(self):
        result = run_cli("params", "--group", "kinetics")
        assert "MuHatHetero20" in result.stdout
        assert "3.2" in result.stdout
        assert "Ks20" in result.stdout
        assert "Bh20" in result.stdout
        assert "MuHatAuto20" in result.stdout


class TestParamsStoichiometry:
    """uct-asp params --group stoichiometry -- display stoichiometric defaults."""

    def test_params_stoichiometry_exits_zero(self):
        result = run_cli("params", "--group", "stoichiometry")
        assert result.returncode == 0

    def test_params_stoichiometry_shows_header(self):
        result = run_cli("params", "--group", "stoichiometry")
        assert "StoichiometricParams" in result.stdout

    def test_params_stoichiometry_shows_key_params(self):
        result = run_cli("params", "--group", "stoichiometry")
        assert "Yh" in result.stdout
        assert "0.666" in result.stdout
        assert "Ya" in result.stdout
        assert "CODVSS" in result.stdout


class TestParamsWastewater:
    """uct-asp params --group wastewater -- display wastewater defaults."""

    def test_params_wastewater_exits_zero(self):
        result = run_cli("params", "--group", "wastewater")
        assert result.returncode == 0

    def test_params_wastewater_shows_key_params(self):
        result = run_cli("params", "--group", "wastewater")
        assert "Sti" in result.stdout
        assert "500" in result.stdout
        assert "Nti" in result.stdout


class TestParamsPlant:
    """uct-asp params --group plant -- display plant config defaults."""

    def test_params_plant_exits_zero(self):
        result = run_cli("params", "--group", "plant")
        assert result.returncode == 0

    def test_params_plant_shows_key_fields(self):
        result = run_cli("params", "--group", "plant")
        assert "PlantConfig" in result.stdout
        assert "LastReactor" in result.stdout
        assert "FlowFeed" in result.stdout


class TestParamsIntegration:
    """uct-asp params --group integration -- display integration defaults."""

    def test_params_integration_exits_zero(self):
        result = run_cli("params", "--group", "integration")
        assert result.returncode == 0

    def test_params_integration_shows_key_fields(self):
        result = run_cli("params", "--group", "integration")
        assert "IntegrationParams" in result.stdout
        assert "Accuracy" in result.stdout


class TestParamsNoGroup:
    """uct-asp params without --group -- should print guidance and exit 1."""

    def test_params_no_group_exits_nonzero(self):
        result = run_cli("params")
        assert result.returncode == 1

    def test_params_no_group_shows_guidance(self):
        result = run_cli("params")
        assert "--group" in result.stdout


class TestParamsInvalidGroup:
    """uct-asp params --group invalid -- argparse rejects it with exit 2."""

    def test_params_invalid_group_exits_nonzero(self):
        result = run_cli("params", "--group", "nonexistent")
        assert result.returncode != 0


class TestSteadyStateNoConfig:
    """uct-asp steady-state with no --config -- defaults have Rs=0, should fail."""

    def test_steady_state_no_config_exits_nonzero(self):
        """With default params (Rs=0), should fail with validation error."""
        result = run_cli("steady-state")
        assert result.returncode != 0

    def test_steady_state_no_config_error_message(self):
        result = run_cli("steady-state")
        assert "Error" in result.stderr or "error" in result.stderr.lower()
        assert "Rs" in result.stderr


class TestSteadyStateBadConfigPath:
    """uct-asp steady-state --config nonexistent.toml -- error handling."""

    def test_bad_config_path_exits_nonzero(self):
        result = run_cli("steady-state", "--config", "/nonexistent/path/foo.toml")
        assert result.returncode != 0

    def test_bad_config_path_error_message(self):
        result = run_cli("steady-state", "--config", "/nonexistent/path/foo.toml")
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestSteadyStateDefaultPlantToml:
    """uct-asp steady-state --config examples/default_plant.toml -- runs successfully.

    The shipped default_plant.toml has Rs=20.0 and FlowFeed=25.0, so it should
    produce a valid simulation.
    """

    def test_default_plant_config_exits_zero(self):
        result = run_cli("steady-state", "--config", EXAMPLES_CONFIG, timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_default_plant_config_shows_results(self):
        result = run_cli("steady-state", "--config", EXAMPLES_CONFIG, timeout=120)
        assert "STEADY STATE RESULTS" in result.stdout


class TestSteadyStateValidConfig:
    """uct-asp steady-state --config test_plant.toml -- full simulation run."""

    def test_steady_state_exits_zero(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_steady_state_shows_results(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
        assert "STEADY STATE RESULTS" in result.stdout

    def test_steady_state_shows_compound_names(self):
        result = run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)
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
    """uct-asp steady-state --help -- check subcommand help."""

    def test_steady_state_help_exits_zero(self):
        result = run_cli("steady-state", "--help")
        assert result.returncode == 0

    def test_steady_state_help_mentions_config(self):
        result = run_cli("steady-state", "--help")
        assert "--config" in result.stdout


class TestDiurnalSubcommandHelp:
    """uct-asp diurnal --help -- check subcommand help."""

    def test_diurnal_help_exits_zero(self):
        result = run_cli("diurnal", "--help")
        assert result.returncode == 0

    def test_diurnal_help_mentions_config(self):
        result = run_cli("diurnal", "--help")
        assert "--config" in result.stdout
        assert "--diurnal-data" in result.stdout
