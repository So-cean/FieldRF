"""Validate the FieldRF reproduction inputs before training; see --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fieldrf.cli.check_pipeline import main

if __name__ == "__main__":
    main()
