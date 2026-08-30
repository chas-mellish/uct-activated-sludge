"""Unit tests for the CLI module."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uct_activated_sludge.cli import _NumpyEncoder, _build_parser, main


class TestBuildParser:
    """Verify parser structure and argument registration."""

    def test_prog_name(self):
        parser = _build_parser()
        assert parser.prog == "uct-asp"

    def test_subcommands_registered(self):
        parser = _build_parser()
        actions = {a.dest: a for a in parser._subparsers._actions}
        choices = actions["command"].choices
        assert "steady-state" in choices
        assert "diurnal" in choices
        assert "params" in choices

    def test_params_list_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["params", "--list", "kinetics"])
        assert args.list == "kinetics"

    def test_params_list_choices(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["params", "--list", "invalid"])

    def test_steady_state_config_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["steady-state", "--config", "foo.toml"])
        assert args.config == "foo.toml"

    def test_diurnal_data_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["diurnal", "--diurnal-data", "data.csv"])
        assert args.diurnal_data == "data.csv"


class TestMainNoArgs:
    """main() with no args should print help and return 0."""

    def test_no_args_returns_zero(self):
        assert main([]) == 0

    def test_no_args_prints_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "steady-state" in captured.out


class TestMainParams:
    """main() with params subcommand."""

    def test_params_no_list_returns_one(self):
        assert main(["params"]) == 1

    def test_params_no_list_shows_guidance(self, capsys):
        main(["params"])
        captured = capsys.readouterr()
        assert "--list" in captured.out

    def test_params_kinetics_returns_zero(self):
        assert main(["params", "--list", "kinetics"]) == 0

    def test_params_kinetics_shows_header(self, capsys):
        main(["params", "--list", "kinetics"])
        captured = capsys.readouterr()
        assert "KineticParams" in captured.out

    def test_params_stoichiometry_returns_zero(self):
        assert main(["params", "--list", "stoichiometry"]) == 0

    def test_params_all_groups(self):
        for group in ["kinetics", "stoichiometry", "wastewater", "plant", "integration"]:
            assert main(["params", "--list", group]) == 0


class TestSteadyStateCmdErrors:
    """_run_steady_state_cmd error handling."""

    @patch("uct_activated_sludge.config.build_params_from_config")
    def test_value_error_returns_one(self, mock_build, capsys):
        mock_build.side_effect = ValueError("Rs must be > 0")
        rc = main(["steady-state"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Rs must be > 0" in captured.err

    def test_file_not_found_returns_one(self, capsys):
        rc = main(["steady-state", "--config", "/nonexistent/path.toml"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestSteadyStateCmdSuccess:
    """_run_steady_state_cmd with mocked successful simulation."""

    @patch("uct_activated_sludge.steady_state.run_steady_state")
    @patch("uct_activated_sludge.config.build_params_from_config")
    def test_returns_zero_on_success(self, mock_build, mock_run, capsys):
        mock_build.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_run.return_value = {"output": "STEADY STATE RESULTS\nDone."}
        rc = main(["steady-state"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "STEADY STATE RESULTS" in captured.out

    @patch("uct_activated_sludge.steady_state.run_steady_state")
    @patch("uct_activated_sludge.config.build_params_from_config")
    def test_json_output(self, mock_build, mock_run, tmp_path, capsys):
        mock_build.return_value = (
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_run.return_value = {
            "output": "Results",
            "C": np.zeros((4, 15)),
            "C0": np.zeros(15),
            "CSteady": np.zeros((4, 15)),
            "Oc": np.float64(5.0),
            "On": np.float64(3.0),
            "Ot": np.float64(8.0),
            "Denit": np.float64(2.0),
            "converged": True,
            "FlowWaste": np.float64(1.25),
        }
        output_path = tmp_path / "out.json"
        rc = main(["steady-state", "--output", str(output_path)])
        assert rc == 0
        with open(output_path) as f:
            data = json.load(f)
        assert "C" in data
        assert "converged" in data
        assert data["converged"] is True


class TestDiurnalCmdErrors:
    """_run_diurnal_cmd error handling including KeyError and TypeError."""

    def test_file_not_found_returns_one(self, capsys):
        rc = main(["diurnal", "--config", "/nonexistent/path.toml"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    @patch("uct_activated_sludge.config.load_diurnal_data")
    @patch("uct_activated_sludge.steady_state.run_steady_state")
    @patch("uct_activated_sludge.config.build_params_from_config")
    def test_key_error_returns_one(self, mock_build, mock_run, mock_load, capsys):
        mock_build.return_value = (
            MagicMock(), MagicMock(), MagicMock(),
            MagicMock(FlowFeed=25.0), MagicMock(),
        )
        mock_run.return_value = {
            "plant_config": MagicMock(),
            "kinetic_params": MagicMock(),
            "CSteady": MagicMock(),
            "Stoich": MagicMock(),
        }
        mock_load.side_effect = KeyError("Time")
        rc = main(["diurnal", "--diurnal-data", "bad.csv"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Time" in captured.err

    @patch("uct_activated_sludge.diurnal.run_diurnal")
    @patch("uct_activated_sludge.diurnal.default_diurnal_data")
    @patch("uct_activated_sludge.steady_state.run_steady_state")
    @patch("uct_activated_sludge.config.build_params_from_config")
    def test_type_error_returns_one(
        self, mock_build, mock_ss, mock_default_di, mock_run_di, capsys
    ):
        mock_build.return_value = (
            MagicMock(), MagicMock(),
            MagicMock(Sti=500.0, Nti=50.0),
            MagicMock(FlowFeed=25.0), MagicMock(),
        )
        mock_ss.return_value = {
            "plant_config": MagicMock(),
            "kinetic_params": MagicMock(),
            "CSteady": MagicMock(),
            "Stoich": MagicMock(),
        }
        mock_default_di.return_value = [MagicMock()]
        mock_run_di.side_effect = TypeError("bad record type")
        rc = main(["diurnal"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "bad record type" in captured.err


class TestNumpyEncoder:
    """Verify _NumpyEncoder handles numpy types correctly."""

    def test_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = json.loads(json.dumps(arr, cls=_NumpyEncoder))
        assert result == [1.0, 2.0, 3.0]

    def test_float64(self):
        val = np.float64(3.14)
        result = json.loads(json.dumps(val, cls=_NumpyEncoder))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-10

    def test_int64(self):
        val = np.int64(42)
        result = json.loads(json.dumps(val, cls=_NumpyEncoder))
        assert result == 42
        assert isinstance(result, int)

    def test_bool_(self):
        val = np.bool_(True)
        result = json.loads(json.dumps(val, cls=_NumpyEncoder))
        assert result is True

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            json.dumps(object(), cls=_NumpyEncoder)
