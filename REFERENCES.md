# Project acknowledgements and citations

FieldRF builds on the following projects. Please cite their original work when
using the corresponding datasets, models, implementations or evaluation tools.

| Project | Role in this repository | Reference |
|---|---|---|
| [MRIxFields2026](https://github.com/MRIxFields/MRIxFields2026) | Benchmark, dataset layout, original baselines, reference evaluation and submission conventions | MRIxFields, *MRIxFields2026: A Benchmark for Cross-Field MRI Translation and Harmonization*, 2026; [benchmark repository](https://github.com/MRIxFields/MRIxFields2026) |
| [MONAI](https://github.com/Project-MONAI/MONAI) | Medical-image transforms, MAISI network components and ControlNet | MONAI Consortium, [*MONAI: An open-source framework for deep learning in healthcare*](https://arxiv.org/abs/2211.02701); [upstream citation metadata](https://github.com/Project-MONAI/MONAI/blob/dev/CITATION.cff) |
| [Brain-ID](https://github.com/jhuldr/Brain-ID) | Frozen contrast-agnostic anatomical feature extractor | Liu et al., [*Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging*](https://arxiv.org/abs/2311.16914), ECCV 2024; [upstream BibTeX](https://github.com/jhuldr/Brain-ID#citation) |
| [MAISI / NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR) | Foundation VAE and pretrained brain-MRI rectified-flow initialization | Guo et al., [*MAISI: Medical AI for Synthetic Imaging*](https://arxiv.org/abs/2409.11169), WACV 2025 |
| [MAISI-v2 / NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR) | Rectified-flow architecture and timestep remapping | Zhao et al., [*MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss*](https://arxiv.org/abs/2508.05772), AAAI 2026; [upstream citations](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR#6-citation) |
| [SynthSeg](https://github.com/BBillot/SynthSeg) | Anatomical segmentation used by the reference evaluation | [Project publications and citation instructions](https://github.com/BBillot/SynthSeg) |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) | Perceptual evaluation and paired baseline losses | [Original implementation and citation instructions](https://github.com/richzhang/PerceptualSimilarity#citation) |

The maintained MAISI implementation and model-download links are under
[NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR).
The earlier [MONAI MAISI tutorial](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi)
now redirects users there. Brain-ID's earlier `peirong26/Brain-ID` repository
resolves to the maintained `jhuldr/Brain-ID` repository.

## Comparison methods

The MRI-specific implementations under `Baseline/` adapt these methods to the
benchmark. They are not the upstream authors' original MRI experiments.

| Method | Original project or paper |
|---|---|
| CUT and its 3D adaptation | [Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation) |
| CycleGAN | [PyTorch CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| StarGAN v2 | [Official StarGAN v2 implementation](https://github.com/clovaai/stargan-v2) |
| ITTR | [ITTR: Unpaired Image-to-Image Translation with Transformers](https://arxiv.org/abs/2203.16015) |
| NAFNet | [Official NAFNet implementation](https://github.com/megvii-research/NAFNet) |
| Restormer | [Official Restormer implementation](https://github.com/swz30/Restormer) |
| UNSB | [Official Unpaired Neural Schrödinger Bridge implementation](https://github.com/cyclomon/UNSB) |

External source code, datasets and model weights retain their upstream licenses
and citation requirements. This repository does not redistribute those weights.
