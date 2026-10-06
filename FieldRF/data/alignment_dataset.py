"""Cached datasets for endpoint alignment and synthetic alignment experiments.

``UnpairedEndpointAlignmentDataset`` draws independent real source/reference
samples for distribution losses during endpoint-aligner training.
``PairedAlignmentCacheDataset`` loads same-anatomy synthetic-source / real-target
pairs for the auxiliary synthetic-alignment trainer. That auxiliary stage keeps
real prospective pairs for validation. The final paired adaptation stage uses
``PairedFieldDataset`` from ``data.dataset`` and real prospective training pairs.

The synthetic paired dataset supports two layouts:

1. Directory mode::

       SOURCE_ROOT/SOURCE_SPLIT/latent/{MOD}/{FIELD}/*.pt
       SOURCE_ROOT/SOURCE_SPLIT/brainid_feat/{MOD}/{FIELD}/*.b2nd
       TARGET_ROOT/TARGET_SPLIT/latent/{MOD}/{FIELD}/*.pt
       TARGET_ROOT/TARGET_SPLIT/brainid_feat/{MOD}/{FIELD}/*.b2nd

   Source and target files are paired by the final underscore-delimited token
   in the filename, matching the project's prospective-cache convention.

2. CSV manifest mode, recommended for multiple synthetic variants. Required
   columns are ``pair_id``, ``modality``, ``source_field``, ``target_field``,
   ``source_latent``, ``source_feat``, ``target_latent``, and ``target_feat``.
   Relative paths are resolved against the manifest's parent directory.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch

from data.dataset import _load_feats_b2nd, _load_latent
from models.model import get_class_label


_MANIFEST_COLUMNS = {
    "pair_id",
    "modality",
    "source_field",
    "target_field",
    "source_latent",
    "source_feat",
    "target_latent",
    "target_feat",
}


def _subject_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


class PairedAlignmentCacheDataset(torch.utils.data.Dataset):
    """Load same-anatomy source/target latent and Brain-ID cache pairs."""

    def __init__(
        self,
        *,
        modalities,
        source_fields,
        target_fields,
        feat_cache_name="brainid_feat",
        manifest=None,
        source_cache_root=None,
        target_cache_root=None,
        source_split=None,
        target_split=None,
        max_pairs=0,
    ):
        self.modalities = [str(value) for value in modalities]
        self.source_fields = [str(value) for value in source_fields]
        self.target_fields = [str(value) for value in target_fields]
        self.feat_cache_name = str(feat_cache_name)
        self.records = []

        if manifest:
            self._load_manifest(Path(manifest))
        else:
            required = {
                "source_cache_root": source_cache_root,
                "target_cache_root": target_cache_root,
                "source_split": source_split,
                "target_split": target_split,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "Directory-mode paired alignment requires "
                    + ", ".join(missing)
                )
            self._index_directories(
                Path(source_cache_root),
                Path(target_cache_root),
                str(source_split),
                str(target_split),
            )

        if max_pairs and max_pairs > 0:
            self.records = self.records[: int(max_pairs)]
        if not self.records:
            raise FileNotFoundError(
                "No paired alignment cache records were found. Stage-1 "
                "sample-wise endpoint alignment requires same-anatomy "
                "synthetic-source/real-target pairs; unrelated retrospective "
                "subjects must not be paired randomly. Populate "
                "paired_alignment.source_cache_root with latent/Brain-ID "
                "caches or set paired_alignment.manifest. Prospective "
                "validation pairs must not be substituted for training pairs. "
                "To continue ordinary real-only Stage-1 RF training (without "
                "endpoint alignment), use scripts/train.py. To train "
                "real-only unpaired endpoint distribution alignment, use "
                "scripts/train_endpoint_aligner.py with "
                "configs/endpoint_alignment.yaml."
            )

    def _load_manifest(self, manifest: Path) -> None:
        if not manifest.exists():
            raise FileNotFoundError(f"Alignment manifest does not exist: {manifest}")
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(_MANIFEST_COLUMNS - columns)
            if missing:
                raise ValueError(
                    f"Alignment manifest {manifest} is missing columns: {missing}"
                )
            for row in reader:
                modality = row["modality"]
                source_field = row["source_field"]
                target_field = row["target_field"]
                if modality not in self.modalities:
                    continue
                if source_field not in self.source_fields:
                    continue
                if target_field not in self.target_fields:
                    continue
                record = {
                    "pair_id": row["pair_id"],
                    "modality": modality,
                    "source_field": source_field,
                    "target_field": target_field,
                }
                for key in (
                    "source_latent",
                    "source_feat",
                    "target_latent",
                    "target_feat",
                ):
                    path = _resolve_path(row[key], manifest.parent)
                    if not path.exists():
                        raise FileNotFoundError(
                            f"{key} for pair {row['pair_id']} does not exist: {path}"
                        )
                    record[key] = str(path)
                self.records.append(record)

    def _index_directories(
        self,
        source_root: Path,
        target_root: Path,
        source_split: str,
        target_split: str,
    ) -> None:
        for modality in self.modalities:
            for target_field in self.target_fields:
                target_latent_dir = (
                    target_root / target_split / "latent" / modality / target_field
                )
                target_feat_dir = (
                    target_root
                    / target_split
                    / self.feat_cache_name
                    / modality
                    / target_field
                )
                target_latents = {
                    _subject_id(path): path
                    for path in sorted(target_latent_dir.glob("*.pt"))
                }
                target_feats = {
                    _subject_id(path): path
                    for path in sorted(target_feat_dir.glob("*.b2nd"))
                }
                target_ids = set(target_latents) & set(target_feats)

                for source_field in self.source_fields:
                    if source_field == target_field:
                        continue
                    source_latent_dir = (
                        source_root
                        / source_split
                        / "latent"
                        / modality
                        / source_field
                    )
                    source_feat_dir = (
                        source_root
                        / source_split
                        / self.feat_cache_name
                        / modality
                        / source_field
                    )
                    source_feats = {
                        _subject_id(path): path
                        for path in sorted(source_feat_dir.glob("*.b2nd"))
                    }
                    for source_latent in sorted(source_latent_dir.glob("*.pt")):
                        subject = _subject_id(source_latent)
                        if subject not in target_ids or subject not in source_feats:
                            continue
                        self.records.append(
                            {
                                "pair_id": (
                                    f"{modality}:{subject}:"
                                    f"{source_field}->{target_field}"
                                ),
                                "modality": modality,
                                "source_field": source_field,
                                "target_field": target_field,
                                "source_latent": str(source_latent),
                                "source_feat": str(source_feats[subject]),
                                "target_latent": str(target_latents[subject]),
                                "target_feat": str(target_feats[subject]),
                            }
                        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "pair_id": record["pair_id"],
            "modality": record["modality"],
            "source_field": record["source_field"],
            "target_field": record["target_field"],
            "z_src": _load_latent(record["source_latent"]),
            "feats_src": _load_feats_b2nd(record["source_feat"]),
            "source_class": get_class_label(
                record["modality"], record["source_field"]
            ),
            "z_tgt": _load_latent(record["target_latent"]),
            "feats_tgt": _load_feats_b2nd(record["target_feat"]),
            "target_class": get_class_label(
                record["modality"], record["target_field"]
            ),
        }


class UnpairedEndpointAlignmentDataset(torch.utils.data.Dataset):
    """Draw independent real source/reference samples within one modality.

    No filename or subject matching is performed.  Each source record is
    paired at load time with a random reference-field record from the same
    modality, so the resulting batches are suitable only for distribution
    losses (CORAL/MMD/adversarial), never sample-wise reconstruction losses.
    """

    def __init__(
        self,
        *,
        cache_root,
        split,
        modalities,
        source_fields,
        reference_fields,
        feat_cache_name="brainid_feat",
        max_volumes=0,
    ):
        self.cache_root = Path(cache_root)
        self.split = str(split)
        self.modalities = [str(value) for value in modalities]
        self.source_fields = [str(value) for value in source_fields]
        self.reference_fields = [str(value) for value in reference_fields]
        self.feat_cache_name = str(feat_cache_name)
        self.source_records = []
        self.reference_records = {}

        for modality in self.modalities:
            references = []
            for field in self.reference_fields:
                references.extend(
                    self._records_for(modality, field, max_volumes)
                )
            if not references:
                raise FileNotFoundError(
                    "No unpaired endpoint reference cache records for "
                    f"{modality} fields={self.reference_fields} under "
                    f"{self.cache_root / self.split}"
                )
            self.reference_records[modality] = references

            for field in self.source_fields:
                if field in self.reference_fields:
                    continue
                self.source_records.extend(
                    self._records_for(modality, field, max_volumes)
                )

        if not self.source_records:
            raise FileNotFoundError(
                "No unpaired endpoint source cache records were found under "
                f"{self.cache_root / self.split}. This mode reads only real "
                "latent/Brain-ID caches and does not require synthetic data."
            )

    def _records_for(self, modality, field, max_volumes):
        latent_dir = (
            self.cache_root / self.split / "latent" / modality / field
        )
        feat_dir = (
            self.cache_root
            / self.split
            / self.feat_cache_name
            / modality
            / field
        )
        latent_paths = sorted(latent_dir.glob("*.pt"))
        if max_volumes and max_volumes > 0:
            latent_paths = latent_paths[: int(max_volumes)]

        records = []
        for latent_path in latent_paths:
            feat_path = feat_dir / f"{latent_path.stem}.b2nd"
            if not feat_path.is_file() or feat_path.stat().st_size <= 0:
                continue
            records.append(
                {
                    "modality": modality,
                    "field": field,
                    "latent": str(latent_path),
                    "feat": str(feat_path),
                    "sample_id": latent_path.stem,
                }
            )
        return records

    def __len__(self):
        return len(self.source_records)

    def __getitem__(self, index):
        source = self.source_records[index]
        reference = random.choice(
            self.reference_records[source["modality"]]
        )
        return {
            "modality": source["modality"],
            "source_field": source["field"],
            "reference_field": reference["field"],
            "source_id": source["sample_id"],
            "reference_id": reference["sample_id"],
            "z_src": _load_latent(source["latent"]),
            "feats_src": _load_feats_b2nd(source["feat"]),
            "source_class": get_class_label(
                source["modality"], source["field"]
            ),
            "z_ref": _load_latent(reference["latent"]),
            "feats_ref": _load_feats_b2nd(reference["feat"]),
            "reference_class": get_class_label(
                source["modality"], reference["field"]
            ),
        }
