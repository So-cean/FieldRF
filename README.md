# FieldRF

Anatomy-conditioned 3D rectified flow for cross-field MRI harmonization.
FieldRF uses one shared model across **T1W, T2W and T2FLAIR** and five field
strengths: **0.1T, 1.5T, 3T, 5T and 7T**.

The pipeline combines a MAISI latent representation, frozen Brain-ID anatomical
features, a shared rectified-flow network and paired cross-field adaptation.

| Directory | Contents |
|---|---|
| [FieldRF/](FieldRF/README.md) | Models, data loaders, configurations, training, inference and tests |
| [Baseline/](Baseline/README.md) | CUT, CycleGAN, StarGANv2 and other comparison methods |
| [Evaluation/](Evaluation/README.md) | Reference evaluation implementation |
| [Submission/](Submission/README.md) | Validation submission format and packaging |
| [mrix2026-cli/](mrix2026-cli/README.md) | Segmentation, building, checking and ZIP utilities |
| [Tutorial/](Tutorial/README.md) | Introductory baseline notebooks |

## Getting started

1. Configure PyTorch/torchvision for your hardware, then install
   `requirements.txt`; see [setup](docs/README.md).
2. Obtain the MRI dataset, pretrained MAISI weights and Brain-ID implementation
   and weights separately. Model weights and MRI volumes are not bundled here.
3. Set the dataset, cache and checkpoint paths in `FieldRF/configs/`.
4. Follow the [FieldRF reproduction guide](FieldRF/README.md).

For baseline runs, copy `.env.example` to `.env`, configure local paths, and
follow the [baseline guide](Baseline/README.md).

## Tests and experiment reproduction

The [committed CPU test suite](FieldRF/tests/) contains 25 tests across five
files. Install its dependencies and run it from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q FieldRF/tests
```

The tests cover data pairing, RF/spatial losses, sampling/autograd and
checkpoint/EMA loading with synthetic fixtures and small networks. See the
[test inventory](FieldRF/README.md#cpu-regression-tests). Full experiment
reproduction follows the [method guide](FieldRF/README.md).

Training and inference require the external data, cached representations and
weights described in the method guide. A pretrained checkpoint can be loaded
by specifying its existing filename; model parameter keys are unchanged.
Inference uses no test-time augmentation by default.

See [the method guide](FieldRF/README.md) for the pipeline and
[reference evaluation](Evaluation/README.md) for metric definitions.

## License

See [LICENSE](LICENSE).
