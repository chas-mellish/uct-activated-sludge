import argparse
import json
import sys

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to Python builtins."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uct-asp",
        description="UCT Activated Sludge Model — ASM1/BNR simulation engine",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ss_parser = subparsers.add_parser(
        "steady-state", help="Run steady-state simulation"
    )
    ss_parser.add_argument(
        "--config", type=str, default=None, help="Path to TOML configuration file"
    )
    ss_parser.add_argument(
        "--output", type=str, default=None, help="Path to output file (JSON)"
    )

    di_parser = subparsers.add_parser(
        "diurnal", help="Run diurnal dynamic simulation"
    )
    di_parser.add_argument(
        "--config", type=str, default=None, help="Path to TOML configuration file"
    )
    di_parser.add_argument(
        "--output", type=str, default=None, help="Path to output file (JSON)"
    )
    di_parser.add_argument(
        "--diurnal-data",
        type=str,
        default=None,
        help="Path to diurnal input data file (CSV or TOML)",
    )

    params_parser = subparsers.add_parser(
        "params", help="Display parameter sets"
    )
    params_parser.add_argument(
        "--group",
        type=str,
        choices=["kinetics", "stoichiometry", "wastewater", "plant", "integration"],
        default=None,
        help="Parameter group to display",
    )

    return parser


def _get_version() -> str:
    from uct_activated_sludge import __version__

    return __version__


def _load_config_or_defaults(config_path: str | None) -> dict:
    """Load TOML config from *config_path*, or return an empty dict for defaults."""
    if config_path is not None:
        from uct_activated_sludge.config import load_config

        return load_config(config_path)
    return {}


def _run_steady_state_cmd(args: argparse.Namespace) -> int:
    """Execute the steady-state subcommand."""
    from uct_activated_sludge.config import build_params_from_config
    from uct_activated_sludge.steady_state import run_steady_state

    try:
        config = _load_config_or_defaults(args.config)
        kp, sp, wp, pc, ip = build_params_from_config(config)
        result = run_steady_state(pc, kp, sp, wp, integration_params=ip)

        # Print formatted text output
        print(result["output"])

        # Write JSON output if requested
        if args.output is not None:
            json_result = {
                "C": result["C"],
                "C0": result["C0"],
                "CSteady": result["CSteady"],
                "Oc": result["Oc"],
                "On": result["On"],
                "Ot": result["Ot"],
                "Denit": result["Denit"],
                "converged": result["converged"],
                "FlowWaste": result["FlowWaste"],
            }
            with open(args.output, "w") as f:
                json.dump(json_result, f, cls=_NumpyEncoder, indent=2)
            print(f"Results written to {args.output}")

        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run_diurnal_cmd(args: argparse.Namespace) -> int:
    """Execute the diurnal subcommand."""
    from uct_activated_sludge.config import build_params_from_config, load_diurnal_data
    from uct_activated_sludge.diurnal import default_diurnal_data, run_diurnal
    from uct_activated_sludge.steady_state import run_steady_state

    try:
        config = _load_config_or_defaults(args.config)
        kp, sp, wp, pc, ip = build_params_from_config(config)

        # Run steady-state first to get initial conditions
        ss_result = run_steady_state(pc, kp, sp, wp, integration_params=ip)
        print("Steady-state simulation complete.")

        # Load diurnal data
        if args.diurnal_data is not None:
            diurnal_data = load_diurnal_data(args.diurnal_data)
        else:
            diurnal_data = default_diurnal_data(
                flow=pc.FlowFeed, cod=wp.Sti, tkn=wp.Nti,
            )

        # Run diurnal simulation
        di_result = run_diurnal(
            plant_config=ss_result["plant_config"],
            kinetic_params=ss_result["kinetic_params"],
            stoich_params=sp,
            ww_params=wp,
            integration_params=ip,
            C_steady=ss_result["CSteady"],
            diurnal_data=diurnal_data,
            stoich_matrix=ss_result["Stoich"],
        )

        # Print convergence info
        status = "converged" if di_result["converged"] else "did NOT converge"
        print(
            f"Diurnal simulation {status} after "
            f"{di_result['cycle_count']} cycle(s)."
        )
        print(
            f"Data points per day: {di_result['data_per_day']}  "
            f"(interval: {di_result['data_int_hours']} h)"
        )

        # Write JSON output if requested
        if args.output is not None:
            json_result = {
                "cycle_count": di_result["cycle_count"],
                "converged": di_result["converged"],
                "data_per_day": di_result["data_per_day"],
                "data_int_hours": di_result["data_int_hours"],
                "C_final": di_result["C_final"],
                "response": di_result["response"],
            }
            with open(args.output, "w") as f:
                json.dump(json_result, f, cls=_NumpyEncoder, indent=2)
            print(f"Results written to {args.output}")

        return 0
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run_params_cmd(args: argparse.Namespace) -> int:
    """Execute the params subcommand."""
    from dataclasses import fields

    from uct_activated_sludge.models import (
        IntegrationParams,
        KineticParams,
        PlantConfig,
        StoichiometricParams,
        WastewaterParams,
    )

    group = args.group
    if group is None:
        print("Please specify a parameter group with --group.")
        print("Choices: kinetics, stoichiometry, wastewater, plant, integration")
        return 1

    group_map = {
        "kinetics": ("KineticParams", KineticParams),
        "stoichiometry": ("StoichiometricParams", StoichiometricParams),
        "wastewater": ("WastewaterParams", WastewaterParams),
        "plant": ("PlantConfig", PlantConfig),
        "integration": ("IntegrationParams", IntegrationParams),
    }

    class_name, cls = group_map[group]
    instance = cls()

    print(f"\n  {class_name} — default values\n")
    print(f"  {'Field':<28s} {'Type':<20s} {'Default'}")
    print(f"  {'─' * 28} {'─' * 20} {'─' * 30}")

    for f in fields(cls):
        value = getattr(instance, f.name)
        type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))

        # For numpy arrays, show shape and dtype instead of the full array
        if isinstance(value, np.ndarray):
            display = f"array(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, float):
            display = f"{value:.6g}"
        else:
            display = repr(value)

        print(f"  {f.name:<28s} {type_name:<20s} {display}")

    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "steady-state":
        return _run_steady_state_cmd(args)

    if args.command == "diurnal":
        return _run_diurnal_cmd(args)

    if args.command == "params":
        return _run_params_cmd(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
