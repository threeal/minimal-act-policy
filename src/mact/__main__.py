import argparse

from .train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="mact")
    parser.add_argument("-v", "--version", action="version", version="0.1.0")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train", help="train ACT model from a dataset")
    train_parser.set_defaults(func=train_model)

    train_parser.add_argument("--eval", action="store_true")
    train_parser.add_argument("--onscreen_render", action="store_true")
    train_parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    train_parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    train_parser.add_argument(
        "--seed", action="store", type=int, help="seed", required=True
    )
    train_parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    train_parser.add_argument(
        "--lr", action="store", type=float, help="lr", required=True
    )

    # for ACT
    train_parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    train_parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    train_parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    train_parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    train_parser.add_argument("--temporal_agg", action="store_true")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
