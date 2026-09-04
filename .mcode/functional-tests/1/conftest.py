"""Shared fixtures for functional CLI tests."""

import json
import os
import subprocess

import pytest

WORKSPACE_DIR = os.environ.get(
    "WORKSPACE_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)
REPO_DIR = os.path.join(WORKSPACE_DIR, "uct-activated-sludge")
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CONFIG = os.path.join(TEST_DIR, "test_plant.toml")
EXAMPLES_CONFIG = os.path.join(REPO_DIR, "examples", "default_plant.toml")

CLI_EXE = os.path.join(REPO_DIR, ".venv", "bin", "uct-asp")


def run_cli(*args, input_text=None, timeout=60):
    """Invoke the uct-asp CLI and capture output."""
    cmd = [CLI_EXE] + list(args)
    return subprocess.run(
        cmd,
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )


@pytest.fixture(scope="session")
def cli_help():
    return run_cli("--help")


@pytest.fixture(scope="session")
def cli_no_args():
    return run_cli()


@pytest.fixture(scope="session")
def cli_version():
    return run_cli("--version")


@pytest.fixture(scope="session")
def cli_params_kinetics():
    return run_cli("params", "--list", "kinetics")


@pytest.fixture(scope="session")
def cli_params_stoichiometry():
    return run_cli("params", "--list", "stoichiometry")


@pytest.fixture(scope="session")
def cli_params_wastewater():
    return run_cli("params", "--list", "wastewater")


@pytest.fixture(scope="session")
def cli_params_plant():
    return run_cli("params", "--list", "plant")


@pytest.fixture(scope="session")
def cli_params_integration():
    return run_cli("params", "--list", "integration")


@pytest.fixture(scope="session")
def cli_params_no_list():
    return run_cli("params")


@pytest.fixture(scope="session")
def cli_steady_state_no_config():
    return run_cli("steady-state")


@pytest.fixture(scope="session")
def cli_steady_state_bad_config():
    return run_cli("steady-state", "--config", "/nonexistent/path/foo.toml")


@pytest.fixture(scope="session")
def cli_steady_state_default_plant():
    return run_cli("steady-state", "--config", EXAMPLES_CONFIG, timeout=120)


@pytest.fixture(scope="session")
def cli_steady_state_valid():
    return run_cli("steady-state", "--config", TEST_CONFIG, timeout=120)


@pytest.fixture(scope="session")
def cli_steady_state_json(tmp_path_factory):
    output_path = str(tmp_path_factory.mktemp("ss_json") / "output.json")
    result = run_cli(
        "steady-state", "--config", TEST_CONFIG,
        "--output", output_path,
        timeout=120,
    )
    data = {}
    if result.returncode == 0 and os.path.exists(output_path):
        with open(output_path) as fh:
            data = json.load(fh)
    return result, output_path, data


@pytest.fixture(scope="session")
def cli_steady_state_help():
    return run_cli("steady-state", "--help")


@pytest.fixture(scope="session")
def cli_diurnal_help():
    return run_cli("diurnal", "--help")
