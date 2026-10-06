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
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    from fieldrf.training.endpoint import run
    from fieldrf.training.distributed import teardown_distributed
    try:
        run(args)
    finally:
        teardown_distributed()
