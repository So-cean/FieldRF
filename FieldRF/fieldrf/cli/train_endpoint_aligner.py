"""Command-line entry point for endpoint training."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/endpoint_alignment.yaml",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--max_volumes", type=int, default=0)
    args = parser.parse_args()

    from fieldrf.training.endpoint import run
    run(args)
