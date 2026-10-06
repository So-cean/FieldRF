"""CPU tests for the task-aligned RF paired fine-tuning contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


FIELDRF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIELDRF_ROOT))

from data.dataset import (  # noqa: E402
    PairedFieldDataset,
)


class PairedFieldDatasetTests(unittest.TestCase):
    def _cache(self, root, modalities, fields, subjects):
        for subject in subjects:
            for modality in modalities:
                for field in fields:
                    stem = f"P_{modality}_{field}_{subject}"
                    latent = root / "paired" / "latent" / modality / field
                    feature = root / "paired" / "brainid_feat" / modality / field
                    latent.mkdir(parents=True, exist_ok=True)
                    feature.mkdir(parents=True, exist_ok=True)
                    (latent / f"{stem}.pt").write_bytes(b"latent")
                    (feature / f"{stem}.b2nd").write_bytes(b"feature")

    def test_task1_contains_only_requested_same_modality_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modalities = ["T1W", "T2W"]
            fields = ["0.1T", "3T", "7T"]
            subjects = ["0006"]
            self._cache(root, modalities, fields, subjects)
            dataset = PairedFieldDataset(
                cache_root=root,
                split="paired",
                modalities=modalities,
                source_fields=["0.1T", "3T"],
                target_fields=["7T"],
                subjects=subjects,
            )

            self.assertEqual(len(dataset), 4)
            self.assertTrue(
                all(
                    record["source_modality"] == record["target_modality"]
                    and record["source_field"] in {"0.1T", "3T"}
                    and record["target_field"] == "7T"
                    for record in dataset.records
                )
            )

    def test_task3_has_180_non_identity_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modalities = ["T1W", "T2W", "T2FLAIR"]
            fields = ["0.1T", "1.5T", "3T", "5T", "7T"]
            subjects = ["0006", "0007", "0009"]
            self._cache(root, modalities, fields, subjects)
            dataset = PairedFieldDataset(
                cache_root=root,
                split="paired",
                modalities=modalities,
                source_fields=fields,
                target_fields=fields,
                subjects=subjects,
                include_identity=False,
            )
            self.assertEqual(len(dataset), 180)
            self.assertTrue(
                all(
                    record["source_modality"] == record["target_modality"]
                    and record["source_field"] != record["target_field"]
                    for record in dataset.records
                )
            )

    def test_missing_domain_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing paired latent"):
                PairedFieldDataset(
                    cache_root=directory,
                    split="paired",
                    modalities=["T1W"],
                    source_fields=["3T"],
                    target_fields=["7T"],
                    subjects=["0006"],
                )


if __name__ == "__main__":
    unittest.main()
