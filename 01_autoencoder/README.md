# 01 · Autoencoder

**Hugging Face model:** https://huggingface.co/iaac-maai/hackathon-team4-resplan-ae

A Conditional Variational Autoencoder (CVAE) trained on residential floor plans from ResPlan. The full description is in the [root README](../README.md).

## Use case

**VAE blend (latent interpolation):** two floor plans are encoded into latent vectors. The steps between the two vectors are decoded into new floor-plan representations, which can be used to explore variations of residential layouts.

## Dataset

| | |
|---|---|
| **Name** | ResPlan |
| **Size** | about 17,000 residential floor plans |
| **Source** | https://github.com/dariabasha/ResPlan (fork of the original release, https://www.kaggle.com/datasets/resplan/resplan) |
| **Licence** | see the [root README](../README.md#data) |
| **Input** | 14 geometry channels, rasterised to 64 × 64 px, plus 3 conditions (net area, area, wall depth) |

The data is not in this repository. Put `ResPlan.pkl` in the `data/` folder at the root.

## How to run

Open [`notebook/02_autoencoder_clean_audited.ipynb`](notebook/02_autoencoder_clean_audited.ipynb) in Jupyter or VS Code and run the cells in order. It covers data preparation, training, evaluation, reconstruction, latent interpolation and the ONNX export.

## Setups compared (same split)

| Model | Latent dimension | Parameters | Test objective | Reconstruction loss | KL divergence |
|---|---:|---:|---:|---:|---:|
| CVAE_z32 | 32 | 1.92M | 205.52 | 205.51 | 7.21 |
| **CVAE_z64** | 64 | 2.32M | **187.84** | 187.82 | 15.88 |
| CVAE_z128 | 128 | 3.11M | 196.96 | 196.94 | 22.79 |

CVAE_z64 has the lowest test objective. A bigger latent space (z128) did not improve the result.

## Results

![Original and reconstructed floor plan](figures/cvae_z64_reconstruction.png)

![CVAE latent interpolation](figures/cvae_z64_latent_blend.png)

More figures (training curves for all three setups) are in the [root README](../README.md#5-visual-results) and in [`figures/`](figures/).

## Limitations

- Reconstructed plans are blurry; doors, windows and walls can be incomplete.
- Generated layouts are not checked against building rules or functional needs.
- The output is a raster image, not editable CAD or BIM geometry.
