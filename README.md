# UNet Denoising Production Refactor

This repository refactors `UNetDenoising (6).ipynb` into a production-style, modular codebase.

## Project Layout

```
configs/
  config.yaml
scripts/
  download.py
  s3_push.py
  s3_pull.py
  train.py
  infer.py
tests/
  test_config.py
  test_storage.py
  test_validation.py
Dockerfile
Makefile
.github/workflows/ci.yml
src/unet_denoising/
  cli.py
  config.py
  logging_utils.py
  exceptions.py
  data/
    io.py
    datasets.py
  models/
    unet.py
  training/
    engine.py
  inference/
    halo.py
  pipelines/
    train_pipeline.py
    infer_pipeline.py
  storage/
    google_drive.py
  validation/
    runtime.py
```

## Key Refactor Decisions

- Notebook logic was split into clear layers: config, data, model, training, inference, pipeline.
- Storage is implemented for **Google Drive mounted directories**, not MongoDB.
- All run outputs are written under:
  - `<google_drive_root>/experiments/<experiment_name>/checkpoints`
  - `<google_drive_root>/experiments/<experiment_name>/runs/train_YYYYMMDD_HHMMSS/*`
  - `<google_drive_root>/experiments/<experiment_name>/runs/infer_YYYYMMDD_HHMMSS/*`

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

For development/CI:

```bash
pip install -e ".[dev]"
```

## Configure

Edit `configs/config.yaml` for your paths and hyperparameters.

## Run

```bash
python -m unet_denoising.cli --config configs/config.yaml download
python -m unet_denoising.cli --config configs/config.yaml s3-push
python -m unet_denoising.cli --config configs/config.yaml s3-pull
python -m unet_denoising.cli --config configs/config.yaml train
python -m unet_denoising.cli --config configs/config.yaml infer
```

Or use shortcuts:

```bash
python scripts/download.py
python scripts/s3_push.py
python scripts/s3_pull.py
python scripts/train.py
python scripts/infer.py
```

`download` flow:
- downloads archive from `dataset.download_url` to `dataset.zip_path`
- extracts into `dataset.extract_dir`
- organizes TIFF files into configured train/val/infer folders under `paths.*`

`s3-push` / `s3-pull` flow:
- uses `s3` config section (LocalStack supported via `endpoint_url`)
- syncs dataset splits:
  - `small_images/noisy`, `small_images/gt`
  - `large_images/noisy`, `large_images/gt`
  - `test_images/noisy`, `test_images/gt`
- syncs experiment artifacts for current `storage.experiment_name`
- writes per-run logs/summaries in `runs/s3_push_*` and `runs/s3_pull_*`

If download fails with TLS certificate errors on macOS:
- run the Python certificate installer (`Install Certificates.command`) for your Python installation
- or set `dataset.ca_cert_path` to your PEM CA bundle
- last resort: set `dataset.verify_ssl: false` (not recommended for normal use)

## Production Controls

- Runtime validations fail fast for invalid config, missing dirs/files, missing checkpoint, and unwritable storage root.
- Per-run artifacts are isolated under timestamped run folders.
- Checkpoints are versioned by epoch in `checkpoints/`.
- CLI exits with concise validation errors for operational failures.

## Lockfile + Build

```bash
make lock
make build
```

Artifacts are written to `dist/` (`.whl` + `.tar.gz`).

## Test

```bash
make test
```

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- install
- pytest
- package build

## Docker

Build:

```bash
make docker-build
```

Run (example):

```bash
docker run --rm \
  -v /absolute/path/to/data:/data \
  -v /absolute/path/to/artifacts:/artifacts \
  unet-denoising-prod:latest \
  --config configs/config.yaml infer
```

## Notes

- Training/inference preserves your halo-based patch approach from the notebook.
- Normalization stats are persisted as JSON and reused at inference time.
- Every run writes a dedicated log file (`train.log` / `infer.log`) and `run_summary.json`.
