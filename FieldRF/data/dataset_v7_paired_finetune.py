"""Prospective same-subject, same-modality pairs for V7 RF fine-tuning.

Task 1 and Task 3 translate field strength without changing modality.  The
dataset therefore makes the evaluated source and target field sets explicit
and cannot silently create cross-modality supervision.
"""

from __future__ import annotations

from pathlib import Path

import blosc2
import numpy as np
import torch

from models.bridge_rf_v7 import get_class_label


blosc2.set_nthreads(1)


def load_latent(path: str | Path) -> torch.Tensor:
    """Load one cached four-channel VAE latent."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "latent" not in payload:
        raise KeyError(f"VAE cache has no latent tensor: {path}")
    latent = payload["latent"]
    if latent.ndim != 4 or latent.shape[0] != 4:
        raise ValueError(f"Invalid VAE latent shape {latent.shape}: {path}")
    return latent


def load_feat4(path: str | Path) -> torch.Tensor:
    """Read only V7 feat4 instead of decompressing all five Brain-ID levels."""
    array = blosc2.open(str(path))
    metadata = array.meta
    keys = list(metadata["keys"])
    shapes = metadata["shapes"]
    if "feat4" not in keys:
        raise KeyError(f"Brain-ID cache has no feat4 tensor: {path}")

    offset = 0
    for key in keys:
        shape = tuple(int(value) for value in shapes[key])
        elements = int(np.prod(shape))
        if key == "feat4":
            values = array[offset : offset + elements]
            feature = torch.from_numpy(values.reshape(shape).copy())
            if feature.ndim != 4 or feature.shape[0] != 64:
                raise ValueError(
                    f"Invalid Brain-ID feat4 shape {feature.shape}: {path}"
                )
            return feature
        offset += elements
    raise RuntimeError(f"Failed to read feat4 from {path}")


class ProspectivePairedFieldDataset(torch.utils.data.Dataset):
    """Evaluated same-subject field mappings with a shared modality."""

    def __init__(
        self,
        *,
        cache_root,
        split,
        modalities,
        source_fields,
        target_fields,
        subjects,
        feat_cache_name="brainid_feat",
        include_identity=False,
    ):
        self.root = Path(cache_root) / str(split)
        self.modalities = tuple(str(value) for value in modalities)
        self.source_fields = tuple(str(value) for value in source_fields)
        self.target_fields = tuple(str(value) for value in target_fields)
        self.subjects = tuple(str(value) for value in subjects)
        self.feat_cache_name = str(feat_cache_name)
        self.include_identity = bool(include_identity)
        if (
            not self.modalities
            or not self.source_fields
            or not self.target_fields
            or not self.subjects
        ):
            raise ValueError(
                "modalities, source_fields, target_fields, and subjects "
                "must be non-empty"
            )

        samples = {}
        self.domains = []
        required_fields = tuple(
            dict.fromkeys((*self.source_fields, *self.target_fields))
        )
        for subject in self.subjects:
            for modality in self.modalities:
                for field in required_fields:
                    stem = f"P_{modality}_{field}_{subject}"
                    latent = (
                        self.root / "latent" / modality / field / f"{stem}.pt"
                    )
                    feature = (
                        self.root
                        / self.feat_cache_name
                        / modality
                        / field
                        / f"{stem}.b2nd"
                    )
                    if not latent.is_file() or latent.stat().st_size <= 0:
                        raise FileNotFoundError(f"Missing paired latent: {latent}")
                    if field in self.source_fields:
                        if not feature.is_file() or feature.stat().st_size <= 0:
                            raise FileNotFoundError(
                                f"Missing paired Brain-ID feat4: {feature}"
                            )
                    samples[(subject, modality, field)] = {
                        "domain_key": f"{subject}:{modality}@{field}",
                        "subject": subject,
                        "modality": modality,
                        "field": field,
                        "latent": str(latent),
                        "feature": str(feature),
                    }
                    self.domains.append(samples[(subject, modality, field)])

        self.records = []
        for subject in self.subjects:
            for modality in self.modalities:
                for source_field in self.source_fields:
                    for target_field in self.target_fields:
                        if not self.include_identity and source_field == target_field:
                            continue
                        source = samples[(subject, modality, source_field)]
                        target = samples[(subject, modality, target_field)]
                        self.records.append(
                            {
                                "pair_id": (
                                    f"{subject}:{modality}@{source_field}->"
                                    f"{modality}@{target_field}"
                                ),
                                "subject": subject,
                                "source_modality": modality,
                                "source_field": source_field,
                                "target_modality": modality,
                                "target_field": target_field,
                                "source_latent": source["latent"],
                                "source_feature": source["feature"],
                                "source_domain_key": source["domain_key"],
                                "target_latent": target["latent"],
                                "target_feature": target["feature"],
                                "target_domain_key": target["domain_key"],
                            }
                        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        source_modality = record["source_modality"]
        target_modality = record["target_modality"]
        return {
            "pair_id": record["pair_id"],
            "subject": record["subject"],
            "source_modality": source_modality,
            "source_field": record["source_field"],
            "source_domain_key": record["source_domain_key"],
            "target_modality": target_modality,
            "target_field": record["target_field"],
            "target_domain_key": record["target_domain_key"],
            "source_latent": load_latent(record["source_latent"]),
            "source_feat4": load_feat4(record["source_feature"]),
            "source_class": get_class_label(source_modality, record["source_field"]),
            "target_latent": load_latent(record["target_latent"]),
            "target_feat4": load_feat4(record["target_feature"]),
            "target_class": get_class_label(target_modality, record["target_field"]),
        }


__all__ = ["ProspectivePairedFieldDataset", "load_feat4", "load_latent"]
