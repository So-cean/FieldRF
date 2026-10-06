"""Compatibility entry point; implementation lives in fieldrf.cli.train_endpoint_aligner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fieldrf.cli.train_endpoint_aligner import main

if __name__ == "__main__":
    main()
