import argparse

from .sim import record_simulation


def main() -> None:
    parser = argparse.ArgumentParser(prog="mact")
    parser.add_argument("-v", "--version", action="version", version="0.1.0")
    subparsers = parser.add_subparsers(required=True)

    sim_parser = subparsers.add_parser("sim", help="simulation specific commands")
    sim_subparsers = sim_parser.add_subparsers(required=True)

    sim_record_parser = sim_subparsers.add_parser(
        "record", help="record episodes from simulation"
    )
    sim_record_parser.set_defaults(func=record_simulation)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
