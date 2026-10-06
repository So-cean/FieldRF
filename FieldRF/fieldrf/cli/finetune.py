"""Command-line entry point for paired training."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_vis", action="store_true")
    args = parser.parse_args()

    from fieldrf.training.paired import run
    run(args)
