"""Tests that verify documentation examples remain valid.

These tests ensure the usage guide, example files, and CLI help text
stay in sync with the actual code as the project evolves.
"""

from pathlib import Path

import pytest

# Resolve paths relative to the project root (not absolute paths)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _PROJECT_ROOT / "examples"


class TestCLIHelpText:
    """Verify CLI help output contains documented subcommands and flags."""

    def test_subcommands_in_help(self, capsys):
        from uct_activated_sludge.cli import main

        main([])
        captured = capsys.readouterr()
        assert "steady-state" in captured.out
        assert "diurnal" in captured.out
        assert "params" in captured.out

    def test_steady_state_flags(self):
        from uct_activated_sludge.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["steady-state", "--config", "test.toml", "--output", "out.json"])
        assert args.config == "test.toml"
        assert args.output == "out.json"

    def test_diurnal_flags(self):
        from uct_activated_sludge.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "diurnal", "--config", "test.toml",
            "--output", "out.json",
            "--diurnal-data", "data.csv",
        ])
        assert args.config == "test.toml"
        assert args.output == "out.json"
        assert args.diurnal_data == "data.csv"

    def test_params_list_choices(self):
        from uct_activated_sludge.cli import _build_parser

        parser = _build_parser()
        for group in ["kinetics", "stoichiometry", "wastewater", "plant", "integration"]:
            args = parser.parse_args(["params", "--list", group])
            assert args.list == group

    def test_version_flag(self):
        from uct_activated_sludge.cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0


class TestExampleTOMLFiles:
    """Verify example TOML configuration files parse without error."""

    def test_default_plant_toml_loads(self):
        from uct_activated_sludge.config import load_config, build_params_from_config

        config = load_config(str(_EXAMPLES_DIR / "default_plant.toml"))
        kp, sp, wp, pc, ip = build_params_from_config(config)
        assert pc.LastReactor == 3
        assert pc.Rs == 20.0
        assert wp.Sti == 500.0

    def test_settled_sewage_toml_loads(self):
        from uct_activated_sludge.config import load_config, build_params_from_config

        config = load_config(str(_EXAMPLES_DIR / "settled_sewage.toml"))
        kp, sp, wp, pc, ip = build_params_from_config(config)
        assert wp.settled is True
        assert wp.Fbs == 0.25
        assert wp.Fup == 0.04
        assert wp.Fus == 0.08

    def test_default_plant_has_all_sections(self):
        from uct_activated_sludge.config import load_config

        config = load_config(str(_EXAMPLES_DIR / "default_plant.toml"))
        for section in ["kinetics", "stoichiometry", "wastewater", "plant", "integration"]:
            assert section in config, f"Missing section: {section}"


class TestExampleCSVFile:
    """Verify the example diurnal CSV file loads without error."""

    def test_diurnal_pattern_csv_loads(self):
        from uct_activated_sludge.config import load_diurnal_data

        data = load_diurnal_data(str(_EXAMPLES_DIR / "diurnal_pattern.csv"))
        assert len(data) == 12

    def test_diurnal_pattern_csv_has_required_keys(self):
        from uct_activated_sludge.config import load_diurnal_data

        data = load_diurnal_data(str(_EXAMPLES_DIR / "diurnal_pattern.csv"))
        for record in data:
            assert "Time" in record
            assert "Flow" in record
            assert "COD" in record
            assert "TKN" in record

    def test_diurnal_pattern_csv_values_positive(self):
        from uct_activated_sludge.config import load_diurnal_data

        data = load_diurnal_data(str(_EXAMPLES_DIR / "diurnal_pattern.csv"))
        for record in data:
            assert record["Flow"] > 0
            assert record["COD"] > 0
            assert record["TKN"] > 0


class TestPythonAPIImports:
    """Verify that documented Python API imports work."""

    def test_import_config_functions(self):
        from uct_activated_sludge.config import (
            load_config,
            build_params_from_config,
            load_diurnal_data,
        )
        assert callable(load_config)
        assert callable(build_params_from_config)
        assert callable(load_diurnal_data)

    def test_import_steady_state(self):
        from uct_activated_sludge.steady_state import run_steady_state

        assert callable(run_steady_state)

    def test_import_diurnal(self):
        from uct_activated_sludge.diurnal import run_diurnal, default_diurnal_data

        assert callable(run_diurnal)
        assert callable(default_diurnal_data)

    def test_import_models(self):
        from uct_activated_sludge.models import (
            KineticParams,
            StoichiometricParams,
            WastewaterParams,
            PlantConfig,
            IntegrationParams,
        )
        assert KineticParams is not None
        assert StoichiometricParams is not None

    def test_import_version(self):
        from uct_activated_sludge import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestPythonAPISnippets:
    """Verify Python API code snippets from the usage guide execute."""

    def test_build_params_from_empty_config(self):
        from uct_activated_sludge.config import build_params_from_config

        kp, sp, wp, pc, ip = build_params_from_config({})
        assert kp.MuHatHetero20 == 3.2
        assert sp.Ya == 0.15
        assert wp.Sti == 500.0

    def test_build_params_from_toml(self):
        from uct_activated_sludge.config import load_config, build_params_from_config

        config = load_config(str(_EXAMPLES_DIR / "default_plant.toml"))
        kp, sp, wp, pc, ip = build_params_from_config(config)
        assert pc.FlowFeed == 25.0
        assert pc.VolumeTotal == 10.5

    def test_inspect_params_with_dataclass_fields(self):
        from dataclasses import fields
        from uct_activated_sludge.models import KineticParams

        kp = KineticParams()
        field_names = [f.name for f in fields(kp)]
        assert "MuHatHetero20" in field_names
        assert "Ks20" in field_names

    def test_stoichiometric_matrix_construction(self):
        from uct_activated_sludge.models import StoichiometricParams
        from uct_activated_sludge.stoichiometry import build_stoichiometric_matrix

        sp = StoichiometricParams()
        matrix = build_stoichiometric_matrix(sp)
        assert matrix.shape[0] > 0
        assert matrix.shape[1] > 0
