"""Cached datasets for shared unpaired RF training and prospective paired adaptation.

Latents use PyTorch dictionaries and Brain-ID features use native blosc2
containers. Class labels share the model's modality-by-field mapping.
"""

from __future__ import annotations


import os
import random
from pathlib import Path

import blosc2
import numpy as np
import torch

from fieldrf.utils.domains import get_class_label  # noqa: E402

# Single-threaded blosc2 per DataLoader worker to avoid oversubscription.
blosc2.set_nthreads(1)

# Decompression params: single-threaded per open() call.
_BLOSC2_DPARAMS = {"nthreads": 1}


def _load_latent(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["latent"]


def _load_feats_b2nd(path):
    """Load Brain-ID features from a native .b2nd file (unified fp16 format).

    The file is a blosc2 container with a 1D fp16 buffer (all feats
    concatenated) + shape metadata in `meta`.  The loader slices the buffer
    by element count and reshapes each feat.
    Returns a list of tensors [feat0, feat1, ..., feat4] (or subset).
    """
    import blosc2

    arr = blosc2.open(str(path))
    meta = arr.meta
    shapes = meta["shapes"]
    keys = meta["keys"]
    raw = arr[:]  # 1D fp16 array (full decompressed buffer)

    feats = []
    offset = 0
    for key in keys:
        shape = shapes[key]
        n_elem = int(np.prod(shape))  # element count (not byte count)
        chunk = raw[offset:offset + n_elem]  # fp16 slice
        feats.append(torch.from_numpy(chunk.reshape(shape).copy()))
        offset += n_elem
    return feats


class MultiFieldCacheDataset(torch.utils.data.Dataset):
    """Multi-modal, multi-field cache dataset for RF.

    Loads latent (4, 96^3) and cached Brain-ID features (feat4 in the supplied
    recipe; multiple levels are supported by the container) for all
    (modality, field) combinations.  Returns joint modality × field class label
    (15-class scheme: 3 mods × 5 fields).

    Stage 1 (graft=False): returns (z, feats, class_label, modality, field) per sample.
    Stage 2 (graft=True): additionally returns a graft pair for adversarial + FM training.

    Graft sampling is controlled by two independent flags:
      - cross_field:     if True, graft tgt field != src field (default: True)
      - cross_modality:  if True, graft tgt modality != src modality (default: True)

    Both default True (any-to-any).  Set cross_modality=False for within-modality
    cross-field only (challenge baseline).  Four combinations are supported:
      (T, T) any-to-any | (T, F) cross-field only | (F, T) cross-modality only | (F, F) identity
    """

    def __init__(self, cache_root, split, modalities, field_names,
                 feat_cache_name="brainid_feat",
                 max_volumes=0, graft=False,
                 cross_field=True, cross_modality=True):
        self.modalities = list(modalities)
        self.field_names = list(field_names)
        self.graft = graft
        self.cross_field = cross_field
        self.cross_modality = cross_modality

        self.files = []            # (latent_path, feat_path)
        self.class_labels = []     # get_class_label(modality, field)
        self.modalities_list = []  # modality string per sample
        self.fields_list = []      # field string per sample

        # Index by (modality, field) for graft sampling.
        self.mod_field_indices = {}  # (mod, field) -> [indices]

        # max_volumes = per (modality, field) combination (0 = no cap)
        per_mod_field = max_volumes if max_volumes > 0 else 0

        for mod in self.modalities:
            for field in self.field_names:
                latent_dir = Path(cache_root) / split / "latent" / mod / str(field)
                feat_dir = Path(cache_root) / split / feat_cache_name / mod / str(field)
                key = (mod, str(field))
                self.mod_field_indices[key] = []
                count = 0
                for pt in sorted(latent_dir.glob("*.pt"))[:per_mod_field if per_mod_field > 0 else None]:
                    stem = pt.stem
                    bl2 = feat_dir / f"{stem}.b2nd"
                    if bl2.exists() and bl2.stat().st_size > 0:
                        idx = len(self.files)
                        self.files.append((str(pt), str(bl2)))
                        self.class_labels.append(get_class_label(mod, str(field)))
                        self.modalities_list.append(mod)
                        self.fields_list.append(str(field))
                        self.mod_field_indices[key].append(idx)
                        count += 1
                print(f"  {mod}/{field}: {count} volumes (class={get_class_label(mod, str(field))})")

        if not self.files:
            raise FileNotFoundError(
                f"No paired files in {cache_root}/{split}/{{{self.modalities}}}/{{{self.field_names}}}"
            )

        if graft:
            for (mod, field), idxs in self.mod_field_indices.items():
                if len(idxs) < 2:
                    print(f"  [graft] WARNING: {mod}/{field} has only {len(idxs)} sample(s); "
                          "graft pairs for this field will reuse the same subject.")

    def __len__(self):
        return len(self.files)

    def _random_other_index(self, mod_field_key, exclude_idx):
        """Pick a random index in (mod, field)'s index list, preferring != exclude_idx."""
        candidates = self.mod_field_indices.get(mod_field_key, [])
        if not candidates:
            return None
        if len(candidates) == 1 or exclude_idx not in candidates:
            return random.choice(candidates)
        opts = [c for c in candidates if c != exclude_idx]
        return random.choice(opts) if opts else random.choice(candidates)

    def __getitem__(self, idx):
        latent_path, feat_path = self.files[idx]
        z = _load_latent(latent_path)
        feats = _load_feats_b2nd(feat_path)  # list of 5 tensors

        out = {
            "z": z,
            "feats": feats,
            "class_label": self.class_labels[idx],
            "modality": self.modalities_list[idx],
            "field_strength": self.fields_list[idx],
            "filename": Path(latent_path).stem,
        }

        if self.graft:
            cur_mod = self.modalities_list[idx]
            cur_field = self.fields_list[idx]

            # ── Choose graft target (modality, field) ──
            if self.cross_modality:
                tgt_mod = random.choice(self.modalities)
            else:
                tgt_mod = cur_mod  # within-modality

            if self.cross_field:
                # Target field can be any field (prefer != current field).
                tgt_candidates = [f for f in self.field_names if f != cur_field] or self.field_names
                tgt_field = random.choice(tgt_candidates)
            else:
                tgt_field = cur_field  # same field (identity on field axis)

            # ── Choose graft source (modality, field) ──
            if self.cross_modality:
                src_mod = random.choice(self.modalities)
            else:
                src_mod = cur_mod  # within-modality

            if self.cross_field:
                src_candidates = [f for f in self.field_names if f != tgt_field] or self.field_names
                src_field = random.choice(src_candidates)
            else:
                src_field = cur_field  # same field

            # graft z_tgt_real: real latent from (tgt_mod, tgt_field), different subject.
            tgt_key = (tgt_mod, tgt_field)
            tgt_j = self._random_other_index(tgt_key, exclude_idx=idx)
            if tgt_j is None:
                graft_z = z.clone()
            else:
                graft_z = _load_latent(self.files[tgt_j][0])

            # graft feats_src: feats from (src_mod, src_field), any subject.
            src_key = (src_mod, src_field)
            src_k = self._random_other_index(src_key, exclude_idx=idx)
            if src_k is None:
                graft_feats = [f.clone() for f in feats]
                graft_z_src = z.clone()
                graft_class_src = self.class_labels[idx]
            else:
                graft_feats = _load_feats_b2nd(self.files[src_k][1])
                # Identity anchor: the same subject's OWN latent (field=src_field),
                # so the trainer can add L1(G(feats_src, class_src), z_src) to keep
                # G honest on the source field during graft adversarial training.
                graft_z_src = _load_latent(self.files[src_k][0])
                graft_class_src = get_class_label(src_mod, src_field)

            out["graft_feats"] = graft_feats
            out["graft_class"] = get_class_label(tgt_mod, tgt_field)
            out["graft_z_tgt_real"] = graft_z
            out["graft_z_src"] = graft_z_src
            out["graft_class_src"] = graft_class_src

        return out


class PairedVisualizationDataset(torch.utils.data.Dataset):
    """Paired prospective data for RF vis (all 5 Brain-ID features)."""

    @staticmethod
    def _subject_id(filename):
        return filename.replace(".b2nd", "").replace(".pt", "").rsplit("_", 1)[-1]

    def __init__(self, latent_root, feat_all_root, source_fields, target_field,
                 modality="T1W", samples_per_field=1):
        target_latent_dir = Path(latent_root) / str(target_field)
        target_feat_dir = Path(feat_all_root) / str(target_field)

        target_latent = {}
        for pt in sorted(target_latent_dir.glob("*.pt")):
            target_latent[self._subject_id(pt.name)] = str(pt)
        target_feat = {}
        for bl2 in sorted(target_feat_dir.glob("*.b2nd")):
            target_feat[self._subject_id(bl2.name)] = str(bl2)

        self.pairs = []
        common_ids = set(target_latent.keys()) & set(target_feat.keys())
        for sf in source_fields:
            src_latent_dir = Path(latent_root) / str(sf)
            src_feat_dir = Path(feat_all_root) / str(sf)

            src_latent = {}
            for pt in sorted(src_latent_dir.glob("*.pt")):
                src_latent[self._subject_id(pt.name)] = str(pt)
            src_feat = {}
            for bl2 in sorted(src_feat_dir.glob("*.b2nd")):
                src_feat[self._subject_id(bl2.name)] = str(bl2)

            count = 0
            for sid in sorted(src_latent.keys()):
                if samples_per_field > 0 and count >= samples_per_field:
                    break
                if sid in common_ids and sid in src_feat:
                    self.pairs.append({
                        "z_src": src_latent[sid],
                        "feats_src": src_feat[sid],
                        "z_tgt": target_latent[sid],
                        "feats_tgt": target_feat[sid],
                        "source_field": str(sf),
                        "target_class": get_class_label(modality, str(target_field)),
                    })
                    count += 1
        if not self.pairs:
            print(f"[PairedVisualizationDataset] WARNING: no paired data")
            self.pairs = []

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return {
            "z_src": _load_latent(p["z_src"]),
            "feats_src": _load_feats_b2nd(p["feats_src"]),
            "z_tgt": _load_latent(p["z_tgt"]),
            "feats_tgt": _load_feats_b2nd(p["feats_tgt"]),
            "source_field": p["source_field"],
            "class_label": p["target_class"],
            "filename": Path(p["z_tgt"]).stem,
        }


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
    """Read only RF feat4 instead of decompressing all five Brain-ID levels."""
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

class PairedFieldDataset(torch.utils.data.Dataset):
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
