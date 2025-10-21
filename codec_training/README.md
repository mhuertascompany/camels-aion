## CAMELS Codec Training Workspace

This directory contains tooling, configuration, and documentation for developing a CAMELS-specific image codec compatible with the Polymathic AION ecosystem.

### Layout

- `configs/` – TOML/JSON configuration files capturing codec architecture, data manifests, and training hyper-parameters.
- `scripts/` – Data preparation and training entrypoints (to be added).
- `logs/` – Default location for TensorBoard or text logs produced during experiments (excluded from version control).
- `checkpoints/` – Staging area for intermediate codec weights before publishing to Hugging Face (excluded from version control).
- `data/` – Local cache for preprocessed training shards (excluded from version control).

### Workflow Overview

1. **Data Preparation**
   ```bash
   python codec_training/scripts/prepare_camels_codec_data.py \
     --output-dir codec_training/data/illustris_codec \
     --suite IllustrisTNG --set LH --redshift 0.0 \
     --normalization-stats path/to/stats.json \
     --batch-size 64 --train-frac 0.9 --val-frac 0.1 --seed 42
   ```
   The command writes split-specific manifests (`train/manifest.json`, `val/manifest.json`, …) plus a `summary.json` describing the dataset.

2. **Codec Training**
   ```bash
   python codec_training/scripts/train_camels_codec.py \
     --train-manifest codec_training/data/illustris_codec/train/manifest.json \
     --val-manifest codec_training/data/illustris_codec/val/manifest.json \
     --codec-repo polymathic-ai/aion-base \
     --output-dir codec_training/checkpoints/camels_legacy_codec \
     --device cuda --epochs 50 --batch-size 32 --grad-accum 2 \
     --lr 3e-4 --scheduler cosine --warmup-epochs 5 --amp
   ```
   Logs are appended to `metrics.jsonl`; checkpoints are stored as `best_codec.pt` (and optionally `last_codec.pt`).

3. **Evaluation**
   - Inspect `metrics.jsonl` for loss trends and codebook usage.
   - Reconstruct held-out shards with the fine-tuned codec and quantify PSNR / SSIM or downstream parameter regression improvements.

4. **Packaging**
   - Mirror the Hugging Face directory structure under `codec_training/checkpoints/<name>/codecs/...`.
   - Update `camels_aion.codec_manager.LocalCodecManager` callers to reference the new snapshot once validated.
