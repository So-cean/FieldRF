# Reproducibility check

## Scope

On 2026-10-06, `So-cean/FieldRF` was cloned from remote `main` into a separate
local directory, starting at commit `6634c88e4c4053ce7b8623cbf3095ad21400531c`.
The fixes and regression tests in this revision were checked there.
No files, data, weights or unpublished helpers from the development checkout
were used as source dependencies.

The check used a separate virtual environment that reused an existing
PyTorch installation and its installed scientific libraries through
`--system-site-packages`. Baseline and the submission CLI were installed into
that virtual environment with `--no-deps --no-build-isolation`. This checks the
released source and installation entry points, rather than a new dependency
resolution from an empty environment. The PyTorch stack was unchanged.

## Tested environment

These are observed versions, not required CUDA/PyTorch pins:

| Component | Version |
|---|---|
| Python | 3.12.9 |
| PyTorch | 2.6.0+cu124; tests executed on CPU |
| torchvision | 0.21.0 |
| MONAI | 1.5.1 |
| NumPy | 2.2.6 |
| blosc2 | 4.7.0 |
| nibabel | 5.3.2 |
| pytest | 9.0.2 |

The runtime dependencies listed in `requirements.txt` were present. Hardware
specific PyTorch/torchvision installation remains the user's choice.

## Verification

```bash
python -m pip install --no-deps --no-build-isolation -e Baseline/ -e mrix2026-cli/
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q FieldRF/tests mrix2026-cli/tests
```

The run passed all 29 tests (27 FieldRF and 2 submission CLI), with 4 subtests.
All 21 source/scorer/CLI entry points passed `--help`. The tests check:

- Cache/script entry points without an external Brain-ID checkout.
- Native `.pt` / `.b2nd` serialization and feature geometry with the real
  MONAI MAISI UNet and ControlNet, a CPU optimization step, checkpoint reload
  with matching predictions and finite RF sampling.
- Subject/modality pairing, RF/alignment/spatial losses, EMA loading and
  Euler/Heun integration and gradients.
- ZIP modality roots, relative output paths and nonzero CLI failure status.

The independent integration check also runs a reduced CUT optimization step,
loads its embedded configuration through the public inference script and
verifies NIfTI shape/affine preservation. Synthetic complete Task 1 and Task 3
prediction trees exercise dry-run, build, source-to-target filename mapping,
axial cropping and ZIP packaging. These checks use synthetic volumes and
reduced network widths; they do not measure MRI image quality.

## Required artifacts for full experiments

| Artifact | Source / configuration |
|---|---|
| MRI dataset and split definitions | [MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026); set `data.data_root` |
| Brain-ID source and pretrained weights | Initialize `third_party/Brain-ID` at `5d8a2d3eef507180b0cbff4cfe418cdfc5c55253`; obtain [Brain-ID weights](https://github.com/jhuldr/Brain-ID/blob/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253/README.md#playing-with-brain-id-synthesizer-and-feature-extractor) and set `brainid.ckpt_path`; `BRAINID_ROOT` is an optional source override |
| MAISI VAE and brain RF weights | Shared VAE from [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT/blob/4156648ce44f5561b75d96b12600fe5afe6b6e8f/models/autoencoder_v1.pt); brain RF from [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain/blob/67ba36a4d3aae5854d7f2f56638e26a641684433/models/diff_unet_3d_rflow-mr-brain_v0.pt); set `vae.checkpoint` and `bridge.ckpt_path` |
| Latent / feature caches | Generate all configured modalities/fields for retrospective, prospective and inference splits; set `data.cache_root` |
| Adapted decoder, endpoint aligner and paired RF | Produced by the [FieldRF stages](FieldRF/README.md#reproduction-pipeline); use the configured epoch selections |
| Task 1/2 segmentations | Separate [SynthSeg](https://github.com/BBillot/SynthSeg) installation; configure `SYNTHSEG_BIN` |

The RF initialization in the supplied recipe is
`diff_unet_3d_rflow-mr-brain_v0.pt`, with `autoencoder_v1.pt`. Record the exact
upstream weight revision and checksums used. Replacing the RF initialization
with another upstream revision changes the experiment. Adaptation and inference
use the same adapted epoch-1 VAE decoder and the selected paired epoch-20 EMA.

The dependency documentation follow-up checked that the clean, pinned public
Brain-ID source imports in the same environment. Its upstream checkpoint loader
also restored a synthetic Brain-ID-format checkpoint containing an
`argparse.Namespace` under PyTorch 2.6 with the documented command-scoped
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` setting. No private Brain-ID source patches
were used. This checks imports and legacy loading; it does not verify the
downloaded pretrained weights or full-volume feature extraction.

Follow the [complete reproduction procedure](FieldRF/README.md#reproduction-pipeline)
for submodule initialization, weight acquisition, YAML paths, both cache writers,
the split requirements and downstream checkpoint selection.

GPU/NPU/DDP execution, full-resolution pretrained Brain-ID/VAE cache generation,
full training and benchmark metric reproduction were not run by this CPU audit.
They require the external artifacts and compute resources above. The repository
does not bundle trained FieldRF weights or a checksum manifest for the original
training artifacts, so exact published metric reproduction is not established
by the passing tests. Reference evaluation implementations are unchanged.

Project attribution and upstream citation instructions are in
[REFERENCES.md](REFERENCES.md).
