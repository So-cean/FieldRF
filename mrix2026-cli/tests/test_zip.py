"""Check the submission ZIP contract through the public CLI."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

CLI_ROOT = Path(__file__).resolve().parents[1]


def test_cli_zip_has_modality_roots_and_supports_relative_output(tmp_path):
    relative = Path("T1W/0.1T_to_7T/pred/P_T1W_7T_0001.nii.gz")
    source = tmp_path / "submission/task1" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic archive payload")
    env = dict(os.environ, PYTHONPATH=str(CLI_ROOT / "src"))
    command = [sys.executable, "-m", "mrix2026_cli", "zip", "--submission-dir", "submission",
               "--output-dir", "zips", "--tasks", "task1"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(tmp_path / "zips/task1.zip") as archive:
        assert archive.namelist() == [relative.as_posix()]
        assert archive.read(relative.as_posix()) == source.read_bytes()


def test_cli_zip_rejects_an_empty_task(tmp_path):
    (tmp_path / "submission/task3").mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(CLI_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "mrix2026_cli", "zip", "--submission-dir", "submission",
         "--output-dir", "zips", "--tasks", "task3"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "no modality files" in result.stderr
    assert not (tmp_path / "zips/task3.zip").exists()
