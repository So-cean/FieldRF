"""Shared constants."""
MODS = ("T1W", "T2W", "T2FLAIR")
TASKS = ("task1", "task2", "task3")

PAIRS = {
    "task1": ["0.1T_to_7T", "1.5T_to_7T", "3T_to_7T", "5T_to_7T"],
    "task2": ["0.1T_to_1.5T", "0.1T_to_3T", "0.1T_to_5T", "0.1T_to_7T"],
    "task3": [f"{s}_to_{t}" for s in ("0.1T","1.5T","3T","5T","7T") for t in ("0.1T","1.5T","3T","5T","7T") if s != t],
}
NEEDS_SEG = {"task1": True, "task2": True, "task3": False}
EXPECTED_FILES_PER_SUBTASK = 3
Z_CLIP_RANGE = (150, 180)
SUBJECT_IDS_BY_SOURCE = {
    "0.1T": ["0001","0002","0003"], "1.5T": ["0004","0005","0008"],
    "3T": ["0010","0011","0012"], "5T": ["0013","0014","0015"],
    "7T": ["0016","0017","0018"],
}
DEFAULT_SWEEP = {
    "task1": ("cut", "pro_pretrained", "epoch100"),
    "task2": ("cut", "pro_pretrained", "epoch100"),
    "task3": ("stargan_v2", "pro_pretrained", "epoch50"),
}

def parse_pair(pair_str: str) -> tuple[str, str]:
    src, tgt = pair_str.split("_to_")
    return src, tgt

def parse_source_dir(name: str) -> tuple[str, str, str]:
    for mod in MODS:
        suffix = f"_{mod}"
        if name.endswith(suffix):
            rest = name[:-len(suffix)]
            for task in TASKS:
                prefix = f"{task}_"
                if rest.startswith(prefix):
                    return task, rest[len(prefix):], mod
    raise ValueError(f"Cannot parse: {name!r}")
