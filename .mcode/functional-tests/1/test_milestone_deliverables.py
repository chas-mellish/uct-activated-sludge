"""Functional tests for M1.1 milestone deliverables.

Verifies the documentation and example files delivered by M1.1:
1. docs/usage-guide.md exists and has expected content
2. README.md exists and has quick-start content
3. examples/diurnal_pattern.csv is loadable by the application
4. examples/settled_sewage.toml is loadable and can drive a simulation
5. tests/test_usage_guide.py documentation validity tests pass
6. tests/conftest.py shared fixtures are present
"""

import csv
import os
import subprocess

WORKSPACE_DIR = os.environ.get(
    "WORKSPACE_DIR",
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )
    ),
)
REPO_DIR = os.path.join(WORKSPACE_DIR, "uct-activated-sludge")


class TestDeliverableFilesExist:
    """Verify all M1.1 deliverable files are present in the repo."""

    def test_usage_guide_exists(self):
        path = os.path.join(REPO_DIR, "docs", "usage-guide.md")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_usage_guide_has_content(self):
        path = os.path.join(REPO_DIR, "docs", "usage-guide.md")
        with open(path) as f:
            content = f.read()
        assert len(content) > 1000, "usage-guide.md is too short"
        assert "uct-asp" in content, "usage-guide.md should reference the CLI"

    def test_readme_exists(self):
        path = os.path.join(REPO_DIR, "README.md")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_readme_has_quickstart(self):
        path = os.path.join(REPO_DIR, "README.md")
        with open(path) as f:
            content = f.read()
        assert len(content) > 200, "README.md is too short for a proper landing page"
        assert "install" in content.lower() or "pip" in content.lower(), (
            "README should mention installation"
        )

    def test_diurnal_pattern_csv_exists(self):
        path = os.path.join(REPO_DIR, "examples", "diurnal_pattern.csv")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_settled_sewage_toml_exists(self):
        path = os.path.join(REPO_DIR, "examples", "settled_sewage.toml")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_test_usage_guide_exists(self):
        path = os.path.join(REPO_DIR, "tests", "test_usage_guide.py")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_conftest_exists(self):
        path = os.path.join(REPO_DIR, "tests", "conftest.py")
        assert os.path.isfile(path), f"Missing: {path}"


class TestDiurnalPatternCSV:
    """Verify examples/diurnal_pattern.csv is loadable by the application."""

    def test_csv_loads_via_app_api(self):
        """Load the CSV using the application's own load_diurnal_data function."""
        from uct_activated_sludge.config import load_diurnal_data

        csv_path = os.path.join(REPO_DIR, "examples", "diurnal_pattern.csv")
        data = load_diurnal_data(csv_path)
        assert len(data) == 12, f"Expected 12 records, got {len(data)}"

    def test_csv_records_have_required_keys(self):
        from uct_activated_sludge.config import load_diurnal_data

        csv_path = os.path.join(REPO_DIR, "examples", "diurnal_pattern.csv")
        data = load_diurnal_data(csv_path)
        for record in data:
            assert "Time" in record, "Missing 'Time' key"
            assert "Flow" in record, "Missing 'Flow' key"
            assert "COD" in record, "Missing 'COD' key"
            assert "TKN" in record, "Missing 'TKN' key"

    def test_csv_values_positive(self):
        from uct_activated_sludge.config import load_diurnal_data

        csv_path = os.path.join(REPO_DIR, "examples", "diurnal_pattern.csv")
        data = load_diurnal_data(csv_path)
        for record in data:
            assert record["Flow"] > 0, f"Flow must be positive, got {record['Flow']}"
            assert record["COD"] > 0, f"COD must be positive, got {record['COD']}"
            assert record["TKN"] > 0, f"TKN must be positive, got {record['TKN']}"

    def test_csv_raw_parse(self):
        """Verify the CSV can be parsed with stdlib csv module too."""
        csv_path = os.path.join(REPO_DIR, "examples", "diurnal_pattern.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 12
        assert "Time" in rows[0]


class TestSettledSewageToml:
    """Verify examples/settled_sewage.toml is loadable and drives a simulation."""

    def test_settled_sewage_config_loads(self):
        from uct_activated_sludge.config import load_config, build_params_from_config

        config_path = os.path.join(REPO_DIR, "examples", "settled_sewage.toml")
        config = load_config(config_path)
        kp, sp, wp, pc, ip = build_params_from_config(config)
        assert wp.settled is True
        assert wp.Fbs == 0.25
        assert wp.Fup == 0.04
        assert wp.Fus == 0.08

    def test_settled_sewage_has_all_sections(self):
        from uct_activated_sludge.config import load_config

        config_path = os.path.join(REPO_DIR, "examples", "settled_sewage.toml")
        config = load_config(config_path)
        for section in ["kinetics", "stoichiometry", "wastewater", "plant", "integration"]:
            assert section in config, f"Missing section: {section}"

    def test_settled_sewage_steady_state_simulation(self):
        """Verify the settled_sewage config can drive a complete simulation."""
        from uct_activated_sludge.config import load_config, build_params_from_config
        from uct_activated_sludge.steady_state import run_steady_state

        config_path = os.path.join(REPO_DIR, "examples", "settled_sewage.toml")
        config = load_config(config_path)
        kp, sp, wp, pc, ip = build_params_from_config(config)
        result = run_steady_state(pc, kp, sp, wp, integration_params=ip)
        assert result["converged"] is True, "Simulation did not converge"
        assert result["FlowWaste"] > 0.0
        assert "STEADY STATE RESULTS" in result["output"]


class TestUsageGuideTestsPass:
    """Run the documentation validity tests (tests/test_usage_guide.py) as subprocess."""

    def test_usage_guide_tests_pass(self):
        """Execute tests/test_usage_guide.py via pytest and verify all pass."""
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_usage_guide.py", "-v", "--tb=short"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"test_usage_guide.py failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
        assert "passed" in result.stdout


class TestConfTestFixtures:
    """Verify tests/conftest.py has the expected shared fixtures."""

    def test_conftest_has_new_fixtures(self):
        """Verify conftest.py includes the fixtures added by M1.1."""
        conftest_path = os.path.join(REPO_DIR, "tests", "conftest.py")
        with open(conftest_path) as f:
            content = f.read()
        # Fixtures added by M1.1
        assert "cli_parser" in content, "Missing cli_parser fixture"
        assert "diurnal_pattern_data" in content, "Missing diurnal_pattern_data fixture"
        assert "default_plant_toml_config" in content, "Missing default_plant_toml_config fixture"
        assert "default_plant_params" in content, "Missing default_plant_params fixture"
        assert "default_kinetic_params" in content, "Missing default_kinetic_params fixture"
        assert "default_stoich_params" in content, "Missing default_stoich_params fixture"
