from __future__ import annotations

import json
import shutil
from pathlib import Path

from unet_denoising.config import AppConfig
from unet_denoising.data.download import download_file, extract_zip, organize_dataset
from unet_denoising.exceptions import ConfigValidationError
from unet_denoising.logging_utils import add_file_handler, get_logger
from unet_denoising.storage.google_drive import GoogleDriveStorage
from unet_denoising.validation.runtime import ensure_writable_dir

logger = get_logger(__name__)


def run_download(cfg: AppConfig) -> None:
    if cfg.dataset is None:
        raise ConfigValidationError("Missing 'dataset' section in config. Add dataset.download_url/zip_path/extract_dir.")

    ensure_writable_dir(Path(cfg.dataset.zip_path).parent, "dataset zip parent")
    ensure_writable_dir(cfg.dataset.extract_dir, "dataset.extract_dir")

    storage = GoogleDriveStorage(Path(cfg.storage.google_drive_root), cfg.storage.experiment_name)
    storage.ensure_dirs()
    run_dir = storage.create_run_dir("download")
    add_file_handler(logger, run_dir / "download.log")

    zip_path = Path(cfg.dataset.zip_path)
    extract_dir = Path(cfg.dataset.extract_dir)

    if cfg.dataset.overwrite and zip_path.exists():
        zip_path.unlink()
    if cfg.dataset.overwrite and extract_dir.exists():
        shutil.rmtree(extract_dir)

    if not zip_path.exists():
        logger.info("Downloading dataset from %s", cfg.dataset.download_url)
        download_file(
            cfg.dataset.download_url,
            zip_path,
            verify_ssl=cfg.dataset.verify_ssl,
            ca_cert_path=cfg.dataset.ca_cert_path,
        )
    else:
        logger.info("Using existing zip: %s", zip_path)

    if not extract_dir.exists() or cfg.dataset.overwrite:
        logger.info("Extracting zip to %s", extract_dir)
        extract_zip(zip_path, extract_dir)
    else:
        logger.info("Using existing extract dir: %s", extract_dir)

    counts = organize_dataset(extract_dir=extract_dir, paths=cfg.paths)
    logger.info("Organized dataset counts: %s", counts)

    summary = {
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "counts": counts,
        "train_noisy_dir": cfg.paths.noisy_train_dir,
        "train_gt_dir": cfg.paths.gt_train_dir,
        "val_noisy_dir": cfg.paths.noisy_val_dir,
        "val_gt_dir": cfg.paths.gt_val_dir,
        "infer_noisy_dir": cfg.paths.noisy_infer_dir,
        "infer_gt_dir": cfg.paths.gt_infer_dir,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Download pipeline complete. Artifacts: %s", run_dir)
