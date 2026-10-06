"""NIfTI I/O utilities for MRIxFields2026.

Data directory structure:
    {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

    Splits: Training_retrospective, Training_prospective,
            Validating_prospective, Testing_prospective
    Modalities: T1W, T2W, T2FLAIR
    Field strengths: 0.1T, 1.5T, 3T, 5T, 7T
"""

from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np


# 5 field strengths
FIELD_STRENGTHS = ["0.1T", "1.5T", "3T", "5T", "7T"]

# 3 modalities
MODALITIES = ["T1W", "T2W", "T2FLAIR"]

# Data splits — directory name → short abbreviation used in CLI args, yaml configs, and slice filenames.
# Two orthogonal dimensions: collection period (retro/pro) × dataset partition (train/val/test).
# Independent from --mode (retro_scratch / pro_scratch / pro_pretrained), which controls pipeline stage.
SPLIT_ABBR = {
    "Training_retrospective": "retro_train",
    "Training_prospective":   "pro_train",
    "Validating_prospective": "pro_val",
    "Testing_prospective":    "pro_test",
}
ABBR_TO_SPLIT = {v: k for k, v in SPLIT_ABBR.items()}
# Domain index mapping for StarGAN v2
FIELD_TO_DOMAIN = {fs: i for i, fs in enumerate(FIELD_STRENGTHS)}


def get_joint_domain(modality: str, field_strength: str) -> int:
    """Map (modality, field_strength) to a flat domain index in [0, NUM_JOINT_DOMAINS)."""
    return MODALITIES.index(modality) * len(FIELD_STRENGTHS) + FIELD_STRENGTHS.index(field_strength)


def load_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (data, affine).

    The image is reoriented to RAS+ canonical form for consistency
    across different scanner conventions.
    """
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    """Save numpy array as a NIfTI file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    nib.save(img, str(path))


def list_nifti_files(
    data_root: str | Path,
    split: str,
    modality: str,
    field_strength: str,
) -> List[Path]:
    """List all NIfTI files for a given split/modality/field_strength.

    Path pattern: {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

    Args:
        data_root: Root data directory (e.g., Data_to_be_released/).
        split: e.g., "Training_retrospective".
        modality: e.g., "T1W".
        field_strength: e.g., "0.1T", "3T".

    Returns:
        Sorted list of .nii.gz file paths.
    """
    root = Path(data_root) / split / modality / field_strength
    if not root.exists():
        return []
    return sorted(root.glob("*.nii.gz"))


def extract_subject_id(filename: str) -> str:
    """Extract subject ID from standardized filename for cross-field pairing.

    Filename format: {R,P}_{modality}_{field}_{ID}.nii.gz
    Examples:
        P_T1W_0.1T_0001.nii.gz -> P_0001
        R_T1W_3T_0042.nii.gz   -> R_0042
    """
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return f"{parts[0]}_{parts[-1]}"
    return base



