"""Environment — load .env from repo root."""
import os, sys
from pathlib import Path

def find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / ".env.example").exists():
            return p
    return cwd

def load_env(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = find_repo_root()
    env_path = repo_root / ".env"
    if not env_path.exists():
        return repo_root
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v
    return repo_root

def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)
