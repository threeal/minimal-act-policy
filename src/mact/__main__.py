import argparse

from .train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="mact")
    parser.add_argument("-v", "--version", action="version", version="0.1.0")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train", help="train ACT model from a dataset")
    train_parser.set_defaults(func=train_model)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
