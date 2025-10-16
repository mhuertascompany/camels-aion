# CAMELS ↔︎ AION Pipeline

This repository contains scripts to encode CAMELS 2D maps with the Polymathic AION model and train lightweight regressors that map AION embeddings to cosmological and astrophysical parameters.

## Prerequisites

- Access to the CAMELS 2D map archive mirroring the official directory layout (e.g. `/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data`).
- A Jean-Zay (or equivalent) environment with Python ≥ 3.10, PyTorch + CUDA, and the `polymathic-aion[torch]` package installed.
- A valid Hugging Face token with access to the gated `polymathic-ai/aion-base` checkpoint (run `huggingface-cli login` beforehand).

Set the base dataset location globally via:

```bash
export CAMELS_BASE_PATH=/lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data
```

## Jean-Zay Environment Setup

The outline below assumes a typical Jean-Zay interactive session (e.g. `srun --pty bash` on a GPU node). Adjust module names if your project uses different versions.

1. **Load base modules**
   ```bash
   module purge
   module load python/3.10 cuda/12.1
   module load gcc/11.3  # only if your project requires a specific toolchain
   ```

2. **Create and activate a virtual environment** *(recommended)*
   ```bash
   python -m venv $WORK/venvs/camels-aion
   source $WORK/venvs/camels-aion/bin/activate
   python -m pip install --upgrade pip wheel setuptools
   ```
   If you prefer to rely solely on Jean-Zay modules, you may skip this step. In that case, remember to install the Python packages in your user site (e.g. `pip install --user polymathic-aion huggingface_hub`) and ensure `$PYTHONPATH` includes this repository when running scripts.

3. **Install dependencies**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121  # pick the CUDA build matching the loaded module
   pip install polymathic-aion[torch] huggingface_hub
   ```
   If your project already ships a compatible PyTorch module, you can skip the explicit `pip install torch` line and rely on the module-provided wheel.

4. **Authenticate with Hugging Face (once per environment)**
   ```bash
   huggingface-cli login --token <HF_TOKEN>
   ```
   Consider setting `HF_HOME` to a project-specific directory on Lustre to avoid repeated downloads:
   ```bash
   export HF_HOME=$WORK/.cache/huggingface
   ```

5. **Mirror the AION model locally (required on compute nodes without network)**
   ```bash
   python scripts/download_aion_model.py --dest $WORK/models/aion
   ```
   This downloads the `polymathic-ai/aion-base` snapshot once (run from a login node with internet access).

6. **Download codec/tokenizer weights**
   ```bash
   python scripts/download_aion_codecs.py --repo $WORK/models/aion
   ```
   (Omit the `--repo` flag to pull directly from Hugging Face.) The codec weights are cached under `$HF_HOME` (or the default Hugging Face cache). Make sure that directory is accessible from compute nodes.

7. **Quick sanity check**
   Run the environment test script to verify PyTorch and the local AION snapshot:
   ```bash
   export PYTHONPATH=$(pwd):${PYTHONPATH}
   python scripts/check_environment.py --device cuda --model-dir $WORK/models/aion --skip-codecs
   ```
   Use `--skip-model` or `--skip-codecs` for a lightweight check (e.g. on a login node without GPU access) or when codecs have not been mirrored locally yet.

## 1. Inspect IllustrisTNG maps

```bash
python scripts/audit_illustris_maps.py
```

This prints the total sample count, image shape, and quick statistics for the four fields we currently stack (`Mstar`, `Mgas`, `T`, `Z`).

## 2. Encode CAMELS maps with AION

### (Optional) Estimate normalization statistics

To make the LegacySurvey tokenizer behave sensibly on CAMELS maps, compute per-field statistics once:

```bash
python scripts/compute_camels_stats.py \
  --base-path /lustre/fsmisc/dataset/CAMELS_Multifield_Dataset/2D_maps/data \
  --suite IllustrisTNG --set LH --redshift 0.0 \
  --output $SCRATCH/camels_aion/stats/illustris_lh.json
```

The JSON contains the `arcsinh` scaling factors and quantiles used to remap each field into the tokenizer's dynamic range.

For alternative experiments, you can generate different stats files without touching the baseline:

- `scripts/compute_camels_stats_log.py` (`cluster/compute_camels_stats_log.sbatch`) – log1p scaling.
- `scripts/compute_camels_stats_linear.py` (`cluster/compute_camels_stats_linear.sbatch`) – simple linear z-score scaling.

Point `--normalization-stats` at the desired JSON when encoding.

```bash
python scripts/encode_illustris_embeddings.py \
  --suite IllustrisTNG \
  --set LH \
  --redshift 0.0 \
  --output-dir /path/to/illustris_embeddings \
  --batch-size 32 \
  --device cuda \
  --num-encoder-tokens 600 \
  --normalization-stats $SCRATCH/camels_aion/stats/illustris_lh.json
```

Key behaviour:

- Maps are stacked into a 4-channel image (channel order `Mstar`, `Mgas`, `T`, `Z`) and encoded with the LegacySurvey image codec.
- During encoding we reuse the codec's DES band names (`DES-G/R/I/Z`) as placeholders for the four CAMELS channels so that the pretrained tokenizer can be applied offline.
- Embeddings + ground-truth labels are saved in shard files (`*.pt`) alongside a manifest JSON.
- Use `--start-index` / `--end-index` to process subsets, and `--fp32` to disable mixed precision if needed.
- `--device` accepts `cuda`, `cpu`, or `auto` (default). Use `auto` when running on login nodes without GPUs; specify `cuda` explicitly for compute-node jobs.
- `--device` accepts `cuda`, `cpu`, or `auto` (default). Use `auto` when running on login nodes without GPUs; specify `cuda` explicitly for compute-node jobs.

Run the same command with `--suite SIMBA` to prepare transfer-evaluation embeddings.

## 3. Train a regression head on IllustrisTNG embeddings

```bash
python scripts/train_parameter_head.py \
  --manifest /path/to/illustris_embeddings/IllustrisTNG_LH_z0p00_manifest.json \
  --shard-dir /path/to/illustris_embeddings \
  --output-dir outputs/illustris_head \
  --hidden-dim 1024 \
  --epochs 50 \
  --batch-size 512 \
  --lr 3e-4 \
  --device cuda
```

- Uses a 70/15/15 train/val/test split (configurable).
- Supports either a linear head (`--hidden-dim` omitted) or a 2-layer MLP.
- Saves the best checkpoint (`best_model.pt`) and a JSON summary with per-parameter RMSE/MAE.
- Writes `test_predictions.csv` pairing ground-truth parameters with the model's predictions for quick plotting.
- The training script automatically averages AION encodings over the token dimension before fitting the regression head.

## 4. Evaluate zero-shot transfer on SIMBA embeddings

```bash
python scripts/evaluate_head_on_simba.py \
  --model outputs/illustris_head/best_model.pt \
  --manifest /path/to/simba_embeddings/SIMBA_LH_z0p00_manifest.json \
  --shard-dir /path/to/simba_embeddings \
  --output outputs/simba_transfer_metrics.json \
  --batch-size 512 \
  --device cuda
```

The resulting JSON contains per-parameter MSE / MAE on the SIMBA suite so we can compare against the Illustris-trained performance.

## Utilities & Modules

- `camels_aion.config`: Paths / defaults centralised here.
- `camels_aion.data`: Dataset loader for CAMELS maps plus convenience statistics.
- `camels_aion.encoding`: Thin wrapper around `AION` + `CodecManager`, handling batching and shard writing.

## Next Steps

- Perform normalization experiments (per-field scaling vs. raw values).
- Extend the codec layer with CAMELS-specific quantizers if LegacySurvey assumptions prove suboptimal.
- Add logging/notebooks to visualise embeddings and regression residuals across cosmological parameter space.

For SIMBA, use the dedicated pipeline script (defaulting to the SIMBA suite):
```bash
sbatch --export=ALL,SUITE=SIMBA,SET_NAME=LH,
EMBED_DIR=$SCRATCH/camels_aion/embeddings/SIMBA_LH,
HEAD_OUT=$SCRATCH/camels_aion/heads/SIMBA_LH,
NORM_STATS=$SCRATCH/camels_aion/stats/simba_lh.json
cluster/pipeline_simba.sbatch
```

Cross-suite evaluation (Illustris head on SIMBA embeddings):
```bash
sbatch --export=ALL,\
MODEL_PATH=$SCRATCH/camels_aion/heads/IllustrisTNG_LH/best_model.pt,\
MANIFEST=$SCRATCH/camels_aion/embeddings/SIMBA_LH/SIMBA_LH_z0p00_manifest.json,\
SHARD_DIR=$SCRATCH/camels_aion/embeddings/SIMBA_LH,\
OUTPUT=$SCRATCH/camels_aion/evals/illustris_to_simba.json\
cluster/evaluate_head_on_simba.sbatch
```

To compare suites in a shared UMAP projection, use `scripts/compare_umap_embeddings.py` or the `cluster/compare_umap_embeddings.sbatch` wrapper.
```bash
python scripts/compare_umap_embeddings.py \
  --ref-manifest $SCRATCH/camels_aion/embeddings/IllustrisTNG_LH/IllustrisTNG_LH_z0p00_manifest.json \
  --ref-shard-dir $SCRATCH/camels_aion/embeddings/IllustrisTNG_LH \
  --target-manifest $SCRATCH/camels_aion/embeddings/SIMBA_LH/SIMBA_LH_z0p00_manifest.json \
  --target-shard-dir $SCRATCH/camels_aion/embeddings/SIMBA_LH \
  --output-dir $SCRATCH/camels_aion/plots/umap_compare
```
