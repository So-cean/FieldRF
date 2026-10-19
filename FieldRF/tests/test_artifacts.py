"""RF/aligner association survives relocation and rejects incorrect artifacts."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from fieldrf.cli.migrate_checkpoint import migrate_checkpoint
from fieldrf.models.endpoint_aligner import EndpointAligner
from fieldrf.training.paired import _load_endpoint_aligner
from fieldrf.utils.artifacts import checkpoint_sha256, validate_rf_base


class ArtifactTests(unittest.TestCase):
    def test_migrated_checkpoint_survives_move_and_restores_exact_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.mkdir()
            base = original / "rf.pt"
            torch.save({"bridge": {"weight": torch.ones(1)}}, base)
            aligner = EndpointAligner(hidden_channels=8, domain_embed_dim=4, num_blocks=1)
            options = {"hidden_channels": 8, "domain_embed_dim": 4, "num_blocks": 1}
            legacy = original / "aligner.pt"
            torch.save({"endpoint_aligner": aligner.state_dict(),
                        "endpoint_aligner_config": options,
                        "base_checkpoint": str(base)}, legacy)
            moved = root / "moved"
            moved.mkdir()
            migrated = moved / "renamed_aligner.pt"
            original_bytes = legacy.read_bytes()
            migrate_checkpoint(legacy, base, migrated, legacy_root=original)
            self.assertEqual(legacy.read_bytes(), original_bytes)
            relocated_base = moved / "renamed_rf.pt"
            shutil.move(str(base), relocated_base)
            # The recorded old file is genuinely absent; there is no symlink fallback.
            self.assertFalse(base.exists())
            restored, _ = _load_endpoint_aligner(migrated, relocated_base, torch.device("cpu"))
            for key, value in aligner.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_checksum_rejects_wrong_content_at_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "same_name.pt"
            base.write_bytes(b"correct RF")
            state = {"base_checkpoint": str(base), "base_checkpoint_sha256": checkpoint_sha256(base)}
            base.write_bytes(b"another RF")
            with self.assertRaisesRegex(ValueError, "different RF base"):
                validate_rf_base(state, base, root=directory)

    def test_joint_paired_restore_has_no_initialization_file_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.pt"
            aligner = EndpointAligner(hidden_channels=8, domain_embed_dim=4, num_blocks=1)
            torch.save({"bridge": {"weight": torch.ones(1)},
                        "endpoint_aligner": aligner.state_dict(),
                        "endpoint_aligner_config": {"hidden_channels": 8,
                            "domain_embed_dim": 4, "num_blocks": 1}}, path)
            restored, _ = _load_endpoint_aligner(path, "/missing/original_rf.pt", torch.device("cpu"), joint_checkpoint=True)
            for key, value in aligner.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_legacy_matches_content_and_requires_original_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old.pt", root / "new.pt"
            old.write_bytes(b"identical original bytes")
            shutil.copyfile(old, new)
            state = {"base_checkpoint": "old.pt"}
            self.assertEqual(validate_rf_base(state, new, root=root), checkpoint_sha256(new))
            old.unlink()
            with self.assertRaisesRegex(ValueError, "filename alone cannot verify"):
                validate_rf_base(state, new, root=root)

    def test_migration_refuses_overwrite_and_wrong_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, wrong = root / "base.pt", root / "wrong.pt"
            base.write_bytes(b"base")
            wrong.write_bytes(b"wrong")
            endpoint = root / "aligner.pt"
            torch.save({"endpoint_aligner": {}, "base_checkpoint": str(base)}, endpoint)
            with self.assertRaisesRegex(ValueError, "already exists"):
                migrate_checkpoint(endpoint, base, endpoint, legacy_root=root)
            destination = root / "portable.pt"
            with self.assertRaisesRegex(ValueError, "different RF base"):
                migrate_checkpoint(endpoint, wrong, destination, legacy_root=root)
            self.assertFalse(destination.exists())

    def test_replacement_is_checked_even_with_stale_filesystem_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.pt"
            path.write_bytes(b"first")
            first = checkpoint_sha256(path)
            old_stat = path.stat()
            path.write_bytes(b"other")
            with patch.object(Path, "stat", return_value=old_stat):
                self.assertNotEqual(checkpoint_sha256(path), first)
