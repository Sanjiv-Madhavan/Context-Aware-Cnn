from __future__ import annotations

import glob
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

from unet_denoising.config import PathsConfig
from unet_denoising.exceptions import ConfigValidationError
from unet_denoising.logging_utils import get_logger

logger = get_logger(__name__)


def _build_ssl_context(verify_ssl: bool, ca_cert_path: str | None) -> ssl.SSLContext:
    if not verify_ssl:
        return ssl._create_unverified_context()
    if ca_cert_path:
        return ssl.create_default_context(cafile=ca_cert_path)
    return ssl.create_default_context()


def download_file(url: str, dest: Path, verify_ssl: bool = True, ca_cert_path: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ssl_ctx = _build_ssl_context(verify_ssl=verify_ssl, ca_cert_path=ca_cert_path)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        if block_num % 256 == 0:
            logger.info("Download progress: %.1f%%", pct)
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        with opener.open(url) as resp, dest.open("wb") as out_f:
            total = int(resp.headers.get("Content-Length", "0"))
            downloaded = 0
            block_size = 64 * 1024
            block_num = 0
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                block_num += 1
                if total > 0:
                    _report(block_num, block_size, total)
    except ssl.SSLError as exc:
        raise ConfigValidationError(
            "TLS verification failed while downloading dataset. "
            "On macOS Python.org builds run 'Install Certificates.command', "
            "or set dataset.ca_cert_path to your CA bundle, "
            "or (last resort) set dataset.verify_ssl=false."
        ) from exc


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def _copy_files(pattern: str, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(glob.glob(pattern, recursive=True))
    for src in srcs:
        shutil.copy2(src, target_dir / Path(src).name)
    return len(srcs)


def organize_dataset(extract_dir: Path, paths: PathsConfig) -> dict[str, int]:
    counts: dict[str, int] = {}

    counts["train_noisy"] = _copy_files(
        str(extract_dir / "**" / "training" / "patches_input" / "*.tif"),
        Path(paths.noisy_train_dir),
    )
    counts["train_gt"] = _copy_files(
        str(extract_dir / "**" / "training" / "patches_gt" / "*.tif"),
        Path(paths.gt_train_dir),
    )
    counts["val_noisy"] = _copy_files(
        str(extract_dir / "**" / "validation" / "large_input" / "*.tif"),
        Path(paths.noisy_val_dir),
    )
    counts["val_gt"] = _copy_files(
        str(extract_dir / "**" / "validation" / "large_gt" / "*.tif"),
        Path(paths.gt_val_dir),
    )

    # Optional test split patterns (depends on upstream archive naming).
    counts["infer_noisy"] = _copy_files(
        str(extract_dir / "**" / "test*" / "*noisy*" / "*.tif"),
        Path(paths.noisy_infer_dir),
    )
    counts["infer_gt"] = _copy_files(
        str(extract_dir / "**" / "test*" / "*gt*" / "*.tif"),
        Path(paths.gt_infer_dir),
    )

    return counts
