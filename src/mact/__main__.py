import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="mact")
    parser.add_argument("-v", "--version", action="version", version="0.1.0")
    parser.parse_args()

    print("work in progress")


if __name__ == "__main__":
    main()
