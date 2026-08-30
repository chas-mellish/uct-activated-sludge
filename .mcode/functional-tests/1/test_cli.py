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

from conftest import run_cli


class TestHelpOutput:
    """uct-asp --help -- verify help text and exit code."""

    def test_help_exits_zero(self, cli_help):
        assert cli_help.returncode == 0

    def test_help_shows_description(self, cli_help):
        assert "ASM1" in cli_help.stdout or "Activated Sludge" in cli_help.stdout

    def test_help_shows_subcommands(self, cli_help):
        assert "steady-state" in cli_help.stdout
        assert "diurnal" in cli_help.stdout
        assert "params" in cli_help.stdout

    def test_help_shows_prog_name(self, cli_help):
        """Verify the prog name is uct-asp (not uct-as)."""
        assert "uct-asp" in cli_help.stdout

    def test_no_args_shows_help(self, cli_no_args):
        """Running uct-asp with no args should show help and exit 0."""
        assert cli_no_args.returncode == 0
        assert "steady-state" in cli_no_args.stdout


class TestVersionOutput:
    """uct-asp --version -- verify version string."""

    def test_version_exits_zero(self, cli_version):
        assert cli_version.returncode == 0

    def test_version_shows_semver(self, cli_version):
        assert "0.1.0" in cli_version.stdout

    def test_version_shows_prog_name(self, cli_version):
        """Verify the version output includes the correct prog name."""
        assert "uct-asp" in cli_version.stdout


class TestParamsKinetics:
    """uct-asp params --list kinetics -- display kinetic parameter defaults."""

    def test_params_kinetics_exits_zero(self, cli_params_kinetics):
        assert cli_params_kinetics.returncode == 0

    def test_params_kinetics_shows_header(self, cli_params_kinetics):
        assert "KineticParams" in cli_params_kinetics.stdout

    def test_params_kinetics_shows_key_params(self, cli_params_kinetics):
        assert "MuHatHetero20" in cli_params_kinetics.stdout
        assert "3.2" in cli_params_kinetics.stdout
        assert "Ks20" in cli_params_kinetics.stdout
        assert "Bh20" in cli_params_kinetics.stdout
        assert "MuHatAuto20" in cli_params_kinetics.stdout


class TestParamsStoichiometry:
    """uct-asp params --list stoichiometry -- display stoichiometric defaults."""

    def test_params_stoichiometry_exits_zero(self, cli_params_stoichiometry):
        assert cli_params_stoichiometry.returncode == 0

    def test_params_stoichiometry_shows_header(self, cli_params_stoichiometry):
        assert "StoichiometricParams" in cli_params_stoichiometry.stdout

    def test_params_stoichiometry_shows_key_params(self, cli_params_stoichiometry):
        assert "Yh" in cli_params_stoichiometry.stdout
        assert "0.666" in cli_params_stoichiometry.stdout
        assert "Ya" in cli_params_stoichiometry.stdout
        assert "CODVSS" in cli_params_stoichiometry.stdout


class TestParamsWastewater:
    """uct-asp params --list wastewater -- display wastewater defaults."""

    def test_params_wastewater_exits_zero(self, cli_params_wastewater):
        assert cli_params_wastewater.returncode == 0

    def test_params_wastewater_shows_key_params(self, cli_params_wastewater):
        assert "Sti" in cli_params_wastewater.stdout
        assert "500" in cli_params_wastewater.stdout
        assert "Nti" in cli_params_wastewater.stdout


class TestParamsPlant:
    """uct-asp params --list plant -- display plant config defaults."""

    def test_params_plant_exits_zero(self, cli_params_plant):
        assert cli_params_plant.returncode == 0

    def test_params_plant_shows_key_fields(self, cli_params_plant):
        assert "PlantConfig" in cli_params_plant.stdout
        assert "LastReactor" in cli_params_plant.stdout
        assert "FlowFeed" in cli_params_plant.stdout


class TestParamsIntegration:
    """uct-asp params --list integration -- display integration defaults."""

    def test_params_integration_exits_zero(self, cli_params_integration):
        assert cli_params_integration.returncode == 0

    def test_params_integration_shows_key_fields(self, cli_params_integration):
        assert "IntegrationParams" in cli_params_integration.stdout
        assert "Accuracy" in cli_params_integration.stdout


class TestParamsNoGroup:
    """uct-asp params without --list -- should print guidance and exit 1."""

    def test_params_no_group_exits_nonzero(self, cli_params_no_list):
        assert cli_params_no_list.returncode == 1

    def test_params_no_group_shows_guidance(self, cli_params_no_list):
        assert "--list" in cli_params_no_list.stdout


class TestParamsInvalidGroup:
    """uct-asp params --list invalid -- argparse rejects it with exit 2."""

    def test_params_invalid_group_exits_nonzero(self):
        result = run_cli("params", "--list", "nonexistent")
        assert result.returncode != 0


class TestSteadyStateNoConfig:
    """uct-asp steady-state with no --config -- defaults have Rs=0, should fail."""

    def test_steady_state_no_config_exits_nonzero(self, cli_steady_state_no_config):
        """With default params (Rs=0), should fail with validation error."""
        assert cli_steady_state_no_config.returncode != 0

    def test_steady_state_no_config_error_message(self, cli_steady_state_no_config):
        assert "Error" in cli_steady_state_no_config.stderr or "error" in cli_steady_state_no_config.stderr.lower()
        assert "Rs" in cli_steady_state_no_config.stderr


class TestSteadyStateBadConfigPath:
    """uct-asp steady-state --config nonexistent.toml -- error handling."""

    def test_bad_config_path_exits_nonzero(self, cli_steady_state_bad_config):
        assert cli_steady_state_bad_config.returncode != 0

    def test_bad_config_path_error_message(self, cli_steady_state_bad_config):
        assert "not found" in cli_steady_state_bad_config.stderr.lower() or "error" in cli_steady_state_bad_config.stderr.lower()


class TestSteadyStateDefaultPlantToml:
    """uct-asp steady-state --config examples/default_plant.toml -- runs successfully.

    The shipped default_plant.toml has Rs=20.0 and FlowFeed=25.0, so it should
    produce a valid simulation.
    """

    def test_default_plant_config_exits_zero(self, cli_steady_state_default_plant):
        assert cli_steady_state_default_plant.returncode == 0, f"stderr: {cli_steady_state_default_plant.stderr}"

    def test_default_plant_config_shows_results(self, cli_steady_state_default_plant):
        assert "STEADY STATE RESULTS" in cli_steady_state_default_plant.stdout


class TestSteadyStateValidConfig:
    """uct-asp steady-state --config test_plant.toml -- full simulation run."""

    def test_steady_state_exits_zero(self, cli_steady_state_valid):
        assert cli_steady_state_valid.returncode == 0, f"stderr: {cli_steady_state_valid.stderr}"

    def test_steady_state_shows_results(self, cli_steady_state_valid):
        assert "STEADY STATE RESULTS" in cli_steady_state_valid.stdout

    def test_steady_state_shows_compound_names(self, cli_steady_state_valid):
        stdout = cli_steady_state_valid.stdout
        assert "Xbh" in stdout or "Heterotrophic" in stdout or "biomass" in stdout.lower()

    def test_steady_state_shows_vss(self, cli_steady_state_valid):
        assert "Volatile SS" in cli_steady_state_valid.stdout or "VSS" in cli_steady_state_valid.stdout

    def test_steady_state_shows_our(self, cli_steady_state_valid):
        assert "OUR" in cli_steady_state_valid.stdout

    def test_steady_state_json_output(self, cli_steady_state_json):
        """Test that --output flag produces valid JSON."""
        result, _output_path, data = cli_steady_state_json
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "C" in data
        assert "C0" in data
        assert "converged" in data
        assert "FlowWaste" in data
        assert "Oc" in data

    def test_steady_state_json_concentrations_positive(self, cli_steady_state_json):
        """Check that simulation produces positive concentrations."""
        result, _output_path, data = cli_steady_state_json
        assert result.returncode == 0, f"stderr: {result.stderr}"
        C = data["C"]
        for k in range(1, 4):  # reactors 1-3
            for i in range(1, 14):  # compounds 1-13
                assert C[k][i] >= 0.0, (
                    f"Negative concentration at reactor {k}, compound {i}: {C[k][i]}"
                )

    def test_steady_state_converged(self, cli_steady_state_json):
        """Check that the Newton solver converged."""
        result, _output_path, data = cli_steady_state_json
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert data["converged"] is True, "Newton solver did not converge"


class TestSteadyStateSubcommandHelp:
    """uct-asp steady-state --help -- check subcommand help."""

    def test_steady_state_help_exits_zero(self, cli_steady_state_help):
        assert cli_steady_state_help.returncode == 0

    def test_steady_state_help_mentions_config(self, cli_steady_state_help):
        assert "--config" in cli_steady_state_help.stdout


class TestDiurnalSubcommandHelp:
    """uct-asp diurnal --help -- check subcommand help."""

    def test_diurnal_help_exits_zero(self, cli_diurnal_help):
        assert cli_diurnal_help.returncode == 0

    def test_diurnal_help_mentions_config(self, cli_diurnal_help):
        assert "--config" in cli_diurnal_help.stdout
        assert "--diurnal-data" in cli_diurnal_help.stdout
