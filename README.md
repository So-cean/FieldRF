# FieldRF

Anatomy-conditioned 3D rectified flow for cross-field MRI harmonization.
FieldRF uses one shared model across **T1W, T2W and T2FLAIR** and five field
strengths: **0.1T, 1.5T, 3T, 5T and 7T**.

The pipeline combines a MAISI latent representation, frozen Brain-ID anatomical
features, a shared rectified-flow network and paired cross-field adaptation.
**The supplied configurations require both MAISI latent caches and Brain-ID
feature caches.** Brain-ID source is a pinned Git submodule; its pretrained
weights and the MAISI weights must be downloaded separately. See the
[complete cache and training procedure](FieldRF/README.md#reproduction-pipeline).

| Directory | Contents |
|---|---|
| [FieldRF/](FieldRF/README.md) | Models, data loaders, configurations, training, inference and tests |
| [Baseline/](Baseline/README.md) | CUT, CycleGAN, StarGANv2 and other comparison methods |
| [Evaluation/](Evaluation/README.md) | Reference evaluation implementation |
| [Submission/](Submission/README.md) | Validation submission format and packaging |
| [mrix2026-cli/](mrix2026-cli/README.md) | Segmentation, building, checking and ZIP utilities |

## Getting started

```bash
git clone --recurse-submodules https://github.com/So-cean/FieldRF.git
cd FieldRF
# For an existing clone, or a clone made without --recurse-submodules:
git submodule update --init --recursive
```

1. Configure PyTorch/torchvision for your hardware, then install
   `requirements.txt`; see [setup](docs/README.md). It includes the Python
   libraries needed by the frozen Brain-ID feature extractor.
2. Obtain the MRI dataset, the original MAISI VAE/RF weights and the
   [Brain-ID pretrained weights](https://github.com/jhuldr/Brain-ID/blob/5d8a2d3eef507180b0cbff4cfe418cdfc5c55253/README.md#playing-with-brain-id-synthesizer-and-feature-extractor).
3. Set the paths in the four `FieldRF/configs/` files as described in the
   [configuration checklist](FieldRF/README.md#2-configure-data-and-artifact-paths).
4. Generate **both caches for all three dataset splits**, then run pretraining,
   endpoint alignment, decoder adaptation and paired adaptation in the
   [reproduction guide](FieldRF/README.md#reproduction-pipeline).

For baseline runs, copy `.env.example` to `.env`, configure local paths, and
follow the [baseline guide](Baseline/README.md).

## Tests and experiment reproduction

The [committed CPU test suite](FieldRF/tests/) contains 27 tests across six
files. Install its dependencies and run it from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q FieldRF/tests mrix2026-cli/tests
```

Two additional CLI tests check submission ZIP layout. The tests cover data
pairing, RF/spatial losses, sampling/autograd, cache generation and a real MONAI
network's training/checkpoint round trip with synthetic fixtures. See the
[test inventory](FieldRF/README.md#cpu-regression-tests). Full experiment
reproduction follows the [method guide](FieldRF/README.md).

Training and inference require the external data, cached representations and
weights described in the method guide. A pretrained checkpoint can be loaded
by specifying its existing filename; model parameter keys are unchanged.
Inference uses no test-time augmentation by default.

See [the method guide](FieldRF/README.md) for the pipeline and
[reference evaluation](Evaluation/README.md) for metric definitions.

## Acknowledgements and citations

The benchmark, baseline and evaluation framework originates from
[MRIxFields/MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026).
FieldRF uses [MONAI](https://github.com/Project-MONAI/MONAI),
[Brain-ID](https://github.com/jhuldr/Brain-ID) and
[MAISI / MAISI-v2](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR).
Please cite the corresponding original works listed in [REFERENCES.md](REFERENCES.md).

The [reproducibility check](REPRODUCIBILITY.md) records the tested environment,
verification scope and external artifacts needed for full experiments.

## License

See [LICENSE](LICENSE).
