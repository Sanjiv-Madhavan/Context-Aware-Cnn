from __future__ import annotations

import json
import os
from pathlib import Path

from unet_denoising.config import AppConfig
from unet_denoising.exceptions import ConfigValidationError
from unet_denoising.logging_utils import add_file_handler, get_logger
from unet_denoising.storage.google_drive import GoogleDriveStorage
from unet_denoising.validation.runtime import ensure_dir_exists, ensure_writable_dir

logger = get_logger(__name__)


def _require_boto3():
    try:
        import boto3
        from botocore.client import Config
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise ConfigValidationError("boto3 is required for s3-push/s3-pull. Install dependencies again.") from exc
    return boto3, Config


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _local_dataset_dirs(cfg: AppConfig) -> dict[str, Path]:
    return {
        "small_images/noisy": Path(cfg.paths.noisy_train_dir),
        "small_images/gt": Path(cfg.paths.gt_train_dir),
        "large_images/noisy": Path(cfg.paths.noisy_val_dir),
        "large_images/gt": Path(cfg.paths.gt_val_dir),
        "test_images/noisy": Path(cfg.paths.noisy_infer_dir),
        "test_images/gt": Path(cfg.paths.gt_infer_dir),
    }


def _artifact_dir(cfg: AppConfig) -> Path:
    return Path(cfg.storage.google_drive_root) / "experiments" / cfg.storage.experiment_name


def _ensure_bucket(client, bucket: str, region: str) -> None:
    buckets = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if bucket in buckets:
        return
    if region == "us-east-1":
        client.create_bucket(Bucket=bucket)
    else:
        client.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})


def _iter_local_files(base_dir: Path):
    for p in base_dir.rglob("*"):
        if p.is_file():
            yield p


def _upload_dir(client, bucket: str, s3_prefix: str, local_dir: Path) -> int:
    count = 0
    for fp in _iter_local_files(local_dir):
        rel = fp.relative_to(local_dir).as_posix()
        key = f"{s3_prefix}/{rel}" if s3_prefix else rel
        client.upload_file(str(fp), bucket, key)
        count += 1
    return count


def _download_prefix(client, bucket: str, s3_prefix: str, local_dir: Path) -> int:
    paginator = client.get_paginator("list_objects_v2")
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(s3_prefix) :].lstrip("/") if s3_prefix else key
            out_path = local_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(out_path))
            count += 1
    return count


def _build_client(cfg: AppConfig):
    if cfg.s3 is None:
        raise ConfigValidationError("Missing 's3' section in config.")
    boto3, BotoConfig = _require_boto3()

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client(
        "s3",
        endpoint_url=cfg.s3.endpoint_url,
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        verify=cfg.s3.verify_ssl,
        config=BotoConfig(s3={"addressing_style": "path"}),
    )


def run_s3_push(cfg: AppConfig) -> None:
    if cfg.s3 is None:
        raise ConfigValidationError("Missing 's3' section in config.")

    ensure_writable_dir(cfg.storage.google_drive_root, "storage.google_drive_root")
    storage = GoogleDriveStorage(Path(cfg.storage.google_drive_root), cfg.storage.experiment_name)
    storage.ensure_dirs()
    run_dir = storage.create_run_dir("s3_push")
    add_file_handler(logger, run_dir / "s3_push.log")

    client = _build_client(cfg)
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    _ensure_bucket(client, cfg.s3.dataset_bucket, region)
    _ensure_bucket(client, cfg.s3.artifacts_bucket, region)

    uploaded: dict[str, int] = {}
    dataset_root = _normalize_prefix(cfg.s3.dataset_prefix)
    for split, local_dir in _local_dataset_dirs(cfg).items():
        ensure_dir_exists(local_dir, f"dataset local dir ({split})")
        prefix = f"{dataset_root}/{split}" if dataset_root else split
        uploaded[f"dataset:{split}"] = _upload_dir(client, cfg.s3.dataset_bucket, prefix, local_dir)

    artifact_dir = _artifact_dir(cfg)
    artifacts_prefix = _normalize_prefix(f"{cfg.s3.artifacts_prefix}/experiments/{cfg.storage.experiment_name}")
    if artifact_dir.exists():
        uploaded["artifacts"] = _upload_dir(client, cfg.s3.artifacts_bucket, artifacts_prefix, artifact_dir)
    else:
        uploaded["artifacts"] = 0
        logger.warning("Artifact dir not found, skipping upload: %s", artifact_dir)

    summary = {
        "endpoint_url": cfg.s3.endpoint_url,
        "dataset_bucket": cfg.s3.dataset_bucket,
        "artifacts_bucket": cfg.s3.artifacts_bucket,
        "uploaded": uploaded,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("S3 push complete. Summary: %s", summary)


def run_s3_pull(cfg: AppConfig) -> None:
    if cfg.s3 is None:
        raise ConfigValidationError("Missing 's3' section in config.")

    ensure_writable_dir(cfg.storage.google_drive_root, "storage.google_drive_root")
    storage = GoogleDriveStorage(Path(cfg.storage.google_drive_root), cfg.storage.experiment_name)
    storage.ensure_dirs()
    run_dir = storage.create_run_dir("s3_pull")
    add_file_handler(logger, run_dir / "s3_pull.log")

    client = _build_client(cfg)

    downloaded: dict[str, int] = {}
    dataset_root = _normalize_prefix(cfg.s3.dataset_prefix)
    for split, local_dir in _local_dataset_dirs(cfg).items():
        local_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset_root}/{split}" if dataset_root else split
        downloaded[f"dataset:{split}"] = _download_prefix(client, cfg.s3.dataset_bucket, prefix, local_dir)

    artifact_dir = _artifact_dir(cfg)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts_prefix = _normalize_prefix(f"{cfg.s3.artifacts_prefix}/experiments/{cfg.storage.experiment_name}")
    downloaded["artifacts"] = _download_prefix(client, cfg.s3.artifacts_bucket, artifacts_prefix, artifact_dir)

    summary = {
        "endpoint_url": cfg.s3.endpoint_url,
        "dataset_bucket": cfg.s3.dataset_bucket,
        "artifacts_bucket": cfg.s3.artifacts_bucket,
        "downloaded": downloaded,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("S3 pull complete. Summary: %s", summary)
