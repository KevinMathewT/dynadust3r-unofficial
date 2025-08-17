# dynadust3r-unofficial

Unofficial reimplementation of **DynaDUSt3R** trained on **Stereo4D**. The Stereo4D paper details a DynaDUSt3R implementation but does **not** release model weights; this repo recreates that training pipeline based on the paper description and public **DUSt3R** code — for research purposes only.

**Links:** [Stereo4D paper (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Jin_Stereo4D_Learning_How_Things_Move_in_3D_from_Internet_Stereo_CVPR_2025_paper.pdf) · [arXiv](https://arxiv.org/abs/2412.09621) · [Project page](https://stereo4d.github.io/) · [Processing code](https://github.com/Stereo4d/stereo4d-code)

**Datasets:** [Stereo4D annotations (GCS)](https://console.cloud.google.com/storage/browser/stereo4d) · [Left-eye perspective (HF)](https://huggingface.co/datasets/KevinMathew/stereo4d-lefteye-perspective) · [Right-eye perspective (HF)](https://huggingface.co/datasets/KevinMathew/stereo4d-righteye-perspective) *(not used in this training)*

---

## table of contents
- quick start
- install
  - build the `curope` cuda op
- datasets
  - stereo4d (download, layout, convert to webdataset)
- training
  - cli basics & config structure
  - single-gpu / multi-gpu
  - checkpoints, logs, viz
- configuration reference
- troubleshooting
- license

---

## quick start

```bash
# clone
git clone https://github.com/KevinMathewT/dynadust3r-unofficial.git
cd dynadust3r-unofficial

# env (poetry or pip)
poetry install
# or (export deps from poetry to requirements and install with pip)
pip install -r <(poetry export -f requirements.txt --without-hashes)

# build the local cuda op (see section below)
export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0+PTX"
pip install -v --no-build-isolation -e models/croco/curope

# prepare stereo4d shards (see section below)
python extras/preprocess_stereo4d.py \
  dataset.stereo4d.path=/data/stereo4d \
  dataset.stereo4d.lefteye_dir=/data/stereo4d/lefteye-perspective \
  dataset.stereo4d.meta_dir=/data/stereo4d/meta

# train (single gpu example)
python -m train data.loader=stereo4d \
  dataset.stereo4d.path=/data/stereo4d \
  dataset.stereo4d.lefteye_dir=/data/stereo4d/lefteye-perspective \
  dataset.stereo4d.meta_dir=/data/stereo4d/meta
```

---

## install

### build the `curope` cuda op

```bash
export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0+PTX"
pip install -v --no-build-isolation -e models/croco/curope
```

Builds `curope` against your current torch install. Make sure you have a CUDA-enabled torch and toolkit; adjust `TORCH_CUDA_ARCH_LIST` to match your GPU (e.g., `7.5;8.0;9.0+PTX`).

---

## datasets

### stereo4d (required for default runs)

**what it is.** Internet VR180 (stereoscopic) videos processed into per-frame camera poses, 3D tracks, and rectification. We train on the **left-eye perspective** clips (512×512 @ ~60° FoV) paired with official `.npz` annotations.

#### what you download
- **annotations (.npz)** from Google Cloud Storage: `gs://stereo4d/{train,test}/*.npz`.
- **left-eye perspective mp4s** from Hugging Face: `KevinMathew/stereo4d-lefteye-perspective` (tar archives of **plain mp4s**, not WebDataset). You’ll convert them to WebDataset with our script in `extras/`.
- **right-eye perspective mp4s** from Hugging Face: `KevinMathew/stereo4d-righteye-perspective` *(not used in this training, listed for completeness).*

#### download: annotations (.npz) from GCS
```bash
# install / init gcloud (linux example)
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init

# single file example
mkdir -p /data/stereo4d/train /data/stereo4d/test
gcloud storage cp gs://stereo4d/train/CMwZrkhQ0ck_130030030.npz /data/stereo4d/train

# full dataset (mirrors gs://stereo4d under /data/) — multi-TB
gsutil -m cp -R gs://stereo4d /data/
```

Each `.npz` contains (clip-level):  
`name` (e.g., `<videoid>_<timestamp>`), `video_id`, `timestamps`, `camera2world` (per-frame), `track_lengths`, `track_indices`, `track_coordinates` (3D tracks), `rectified2rig` (rectification rotation), `fov_bounds` (VR180 intrinsics).

#### download: left-eye perspective mp4s from HF
```bash
git clone https://huggingface.co/datasets/KevinMathew/stereo4d-lefteye-perspective
cd stereo4d-lefteye-perspective

# pull parts and reconstruct tarballs
git lfs pull --include="*.tar.part_*,test_mp4s.tar"
cat train_mp4s.tar.part_* > train_mp4s.tar

# extract mp4s to your data root
mkdir -p /data/stereo4d/lefteye-perspective/train /data/stereo4d/lefteye-perspective/test
tar -xvf train_mp4s.tar -C /data/stereo4d/lefteye-perspective/train
tar -xvf test_mp4s.tar  -C /data/stereo4d/lefteye-perspective/test
```
Files are named like `<videoid>_<timestamp>-left_rectified.mp4`.

#### recommended on-disk layout (before conversion)
```
/data/stereo4d/
  ├── train/*.npz
  ├── test/*.npz
  ├── lefteye-perspective/
  │   ├── train/*.mp4   # <videoid>_<timestamp>-left_rectified.mp4
  │   └── test/*.mp4
  └── meta/
      ├── stereo4d_id_to_time_and_fov_metadata.csv
      └── stats.csv
```

> the `meta/` csvs are used for filtering & timestamp lookup.

---

### convert to **webdataset** shards (required for training)

We merge **mp4** (left-eye perspective) + **npz** annotations per clip into **sequence-level** WebDataset samples with keys:
- `video.mp4` — rectified left-eye video bytes
- `ann.npz` — official annotations for the clip
- `intr.npy` — 3×3 intrinsics matrix computed from frame width + `hfov` (deg)

Output structure:
```
/data/stereo4d/wds/
  ├── train/stereo4d-000000.tar
  ├── train/stereo4d-000001.tar
  ├── ...
  └── test/stereo4d-000000.tar
        stereo4d-idx.json
        key_to_idx.json
```

#### run the preprocessor

```bash
# base invocation with hydra overrides for paths
python extras/preprocess_stereo4d.py \
  dataset.stereo4d.path=/data/stereo4d \
  dataset.stereo4d.lefteye_dir=/data/stereo4d/lefteye-perspective \
  dataset.stereo4d.meta_dir=/data/stereo4d/meta \
  dataset.stereo4d.hfov=60
```

**Important knobs (edit at the top of `extras/preprocess_stereo4d.py`):**
- `SPLIT = "train" | "test"`
- `NUM_WORKERS = 32`
- `MAX_SHARD_SIZE_GB = 50`, `MAX_SAMPLES_PER_SHARD = 8000`

The script will:
1) **Discover sequences** via metadata CSVs, filtering to reasonable motion.
2) **Validate assets**: ensure each `<seq>-left_rectified.mp4` and `<seq>.npz` exist and decode.
3) **Compute intrinsics** `K` from frame width & `hfov`.
4) **Write shards** with a multiprocessing pool (pattern `stereo4d-%06d.tar`).
5) **Index** shards with WIDS → `stereo4d-idx.json` and `key_to_idx.json`.
6) **Verify** random samples (video + ann + intr).

> tip: put TMP/WIDS cache on fast local scratch; adjust envs at the top of the script.

**What the dataloader expects** (`data.loader=stereo4d` → v5 loader):
- WIDS index at `.../wds/{split}/stereo4d-idx.json` and the key map `key_to_idx.json`.
- Each sample contains `video.mp4`, `ann.npz`, `intr.npy`.

---

## training

### cli basics & config structure

Hydra entrypoint: `@hydra.main(config_path="config", config_name="config")`.

Key knobs:
- `data.loader` (default `stereo4d`)
- `data.size`, `data.batch_size`
- `train.iterations`, `train.validation_frequency`
- `logging.use_wandb`
- `sched` (`cosine`, `linear`, `onecycle`, `steplr`, `exponentiallr`, `reducelronplateau`) or leave unset to disable scheduler

Override any leaf via CLI:
```bash
python -m train \
  data.loader=stereo4d \
  dataset.stereo4d.path=/data/stereo4d \
  dataset.stereo4d.lefteye_dir=/data/stereo4d/lefteye-perspective \
  dataset.stereo4d.meta_dir=/data/stereo4d/meta \
  data.batch_size=4 train.iterations=49000 train.validation_frequency=1000 \
  logging.use_wandb=true
```

### multi-gpu
```bash
# accelerate
accelerate launch -m train data.loader=stereo4d ...

# torchrun
torchrun --nproc_per_node=8 -m train data.loader=stereo4d ...

# slurm (template provided)
sbatch extras/train_dynadust3r.sbatch
```

### checkpoints, logs, viz
- enable W&B via `logging.use_wandb=true`.
- periodic visualizations land in the Hydra run dir (`.../valid/...`).
- top-K checkpoints are kept under `.../checkpoints/` with metric+value in filename.

---

## configuration reference
- `config/config.yaml` – top-level defaults (training, data, logging)
- `config/model/dynadust3r.yaml` – model + DUSt3R weights
- `config/criterion/*.yaml` – loss configs
- `config/optim/*.yaml`, `config/sched/*.yaml` – optimizers & schedulers
- `config/dataset/stereo4d.yaml` – set `path`, `lefteye_dir`, `meta_dir`, `hfov`, `max_frame_window`, splits

---

## troubleshooting
- **curope build** → ensure you’re building against the torch in your current venv; rebuild with `--no-build-isolation` (see snippet above).
- **dataset pairing** → clip ids must match exactly: `<videoid>_<timestamp>` for both `.npz` and `-left_rectified.mp4`.
- **WIDS/webdataset performance** → keep cache/tmp on fast local scratch; tune workers & shard sizes.

---

## license
Parts of DUSt3R/CroCo are non-commercial (CC BY-NC-SA 4.0). Check headers under `models/dust3r/*` and `models/croco/*`.
