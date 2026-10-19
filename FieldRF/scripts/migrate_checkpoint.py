"""Create a verified portable endpoint checkpoint; see --help for arguments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fieldrf.cli.migrate_checkpoint import main

if __name__ == "__main__":
    main()
