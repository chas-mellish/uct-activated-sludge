import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uct-as",
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

    params_parser = subparsers.add_parser(
        "params", help="Display parameter sets"
    )
    params_parser.add_argument(
        "--list",
        type=str,
        choices=["kinetics", "stoichiometry", "wastewater", "plant", "integration"],
        default=None,
        help="Parameter group to display",
    )

    return parser


def _get_version() -> str:
    from uct_activated_sludge import __version__

    return __version__


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "steady-state":
        print("Steady-state simulation not yet implemented.")
        return 1

    if args.command == "diurnal":
        print("Diurnal simulation not yet implemented.")
        return 1

    if args.command == "params":
        print("Parameter display not yet implemented.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
