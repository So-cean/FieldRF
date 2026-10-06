"""Independent real source/reference caches for endpoint distribution alignment.

Source and reference samples share a modality, with no subject matching.
Use these samples for distribution losses. Final same-subject RF adaptation
uses ``PairedFieldDataset`` from ``data.dataset``.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from fieldrf.data.dataset import _load_feats_b2nd, _load_latent
from fieldrf.utils.domains import get_class_label


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
