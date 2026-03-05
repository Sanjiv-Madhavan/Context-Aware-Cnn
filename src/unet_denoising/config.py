from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from unet_denoising.exceptions import ConfigValidationError

@dataclass
class PathsConfig:
    noisy_train_dir: str
    gt_train_dir: str
    noisy_val_dir: str
    gt_val_dir: str
    noisy_infer_dir: str
    gt_infer_dir: str


@dataclass
class StorageConfig:
    root_dir: str
    experiment_name: str

    @property
    def google_drive_root(self) -> str:
        # Backward compatibility with older code paths.
        return self.root_dir


@dataclass
class DataConfig:
    patch_size: int
    border_size: int
    crops_per_image: int
    val_ratio: float
    num_workers: int


@dataclass
class TrainConfig:
    seed: int
    epochs: int
    batch_size: int
    lr: float
    checkpoint_every: int


@dataclass
class InferenceConfig:
    checkpoint_name: str


@dataclass
class MlflowConfig:
    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str = "unet-denoising"
    run_name_prefix: str = "run"


@dataclass
class OptunaConfig:
    enabled: bool = False
    n_trials: int = 10
    top_k: int = 3
    study_name: str = "unet_optuna"
    direction: str = "maximize"
    metric_name: str = "val_ssim"
    timeout_sec: int | None = None
    sampler_seed: int = 42
    lr_min: float = 1e-4
    lr_max: float = 1e-3
    batch_size_choices: list[int] = field(default_factory=lambda: [4, 8])
    patch_size_choices: list[int] = field(default_factory=lambda: [128, 256])
    border_size_choices: list[int] = field(default_factory=lambda: [16, 32])
    crops_per_image_choices: list[int] = field(default_factory=lambda: [64, 128, 165])
    epochs_choices: list[int] = field(default_factory=lambda: [50, 100])


@dataclass
class QualityGateConfig:
    enabled: bool = False
    min_val_psnr: float | None = None
    min_val_ssim: float | None = None
    max_val_loss: float | None = None


@dataclass
class DatasetConfig:
    download_url: str
    zip_path: str
    extract_dir: str
    overwrite: bool = False
    verify_ssl: bool = True
    ca_cert_path: str | None = None


@dataclass
class S3Config:
    endpoint_url: str
    dataset_bucket: str
    artifacts_bucket: str
    dataset_prefix: str = "ai4life-mdc25"
    artifacts_prefix: str = "denoisingfyp"
    verify_ssl: bool = False
    auto_pull: bool = False


@dataclass
class AppConfig:
    paths: PathsConfig
    storage: StorageConfig
    data: DataConfig
    train: TrainConfig
    inference: InferenceConfig
    dataset: DatasetConfig | None = None
    s3: S3Config | None = None
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)



def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    payload: dict[str, Any] = _expand_env_in_obj(yaml.safe_load(cfg_path.read_text()))
    dataset_payload = payload.get("dataset")
    s3_payload = payload.get("s3")
    mlflow_payload = payload.get("mlflow", {})
    optuna_payload = _coerce_optuna_payload(payload.get("optuna", {}))
    quality_gate_payload = payload.get("quality_gate", {})
    storage_payload = dict(payload["storage"])
    if "root_dir" not in storage_payload and "google_drive_root" in storage_payload:
        storage_payload["root_dir"] = storage_payload["google_drive_root"]
    storage_payload.pop("google_drive_root", None)

    cfg = AppConfig(
        paths=PathsConfig(**payload["paths"]),
        storage=StorageConfig(**storage_payload),
        data=DataConfig(**payload["data"]),
        train=TrainConfig(**payload["train"]),
        inference=InferenceConfig(**payload["inference"]),
        dataset=DatasetConfig(**dataset_payload) if dataset_payload else None,
        s3=S3Config(**s3_payload) if s3_payload else None,
        mlflow=MlflowConfig(**mlflow_payload),
        optuna=OptunaConfig(**optuna_payload),
        quality_gate=QualityGateConfig(**quality_gate_payload),
    )
    _validate_config(cfg)
    return cfg


def _coerce_optuna_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    out = dict(payload)
    int_fields = {"n_trials", "top_k", "timeout_sec", "sampler_seed"}
    float_fields = {"lr_min", "lr_max"}
    int_list_fields = {
        "batch_size_choices",
        "patch_size_choices",
        "border_size_choices",
        "crops_per_image_choices",
        "epochs_choices",
    }

    for key in int_fields:
        if key in out and out[key] is not None:
            out[key] = int(out[key])
    for key in float_fields:
        if key in out and out[key] is not None:
            out[key] = float(out[key])
    for key in int_list_fields:
        if key in out and out[key] is not None:
            out[key] = [int(v) for v in out[key]]

    return out


def _validate_config(cfg: AppConfig) -> None:
    _validate_no_unexpanded_vars(cfg)

    if cfg.data.patch_size <= 0:
        raise ConfigValidationError("data.patch_size must be > 0")
    if cfg.data.border_size < 0:
        raise ConfigValidationError("data.border_size must be >= 0")
    if cfg.data.num_workers < 0:
        raise ConfigValidationError("data.num_workers must be >= 0")
    if not (0.0 < cfg.data.val_ratio < 1.0):
        raise ConfigValidationError("data.val_ratio must be in (0, 1)")
    if cfg.train.batch_size <= 0:
        raise ConfigValidationError("train.batch_size must be > 0")
    if cfg.train.epochs <= 0:
        raise ConfigValidationError("train.epochs must be > 0")
    if cfg.train.lr <= 0.0:
        raise ConfigValidationError("train.lr must be > 0")
    if cfg.train.checkpoint_every <= 0:
        raise ConfigValidationError("train.checkpoint_every must be > 0")
    if not cfg.storage.experiment_name.strip():
        raise ConfigValidationError("storage.experiment_name must not be empty")
    if not cfg.storage.root_dir.strip():
        raise ConfigValidationError("storage.root_dir must not be empty")
    if cfg.dataset is not None:
        if not cfg.dataset.download_url.strip():
            raise ConfigValidationError("dataset.download_url must not be empty")
        if not cfg.dataset.zip_path.strip():
            raise ConfigValidationError("dataset.zip_path must not be empty")
        if not cfg.dataset.extract_dir.strip():
            raise ConfigValidationError("dataset.extract_dir must not be empty")
        if cfg.dataset.ca_cert_path is not None and not str(cfg.dataset.ca_cert_path).strip():
            raise ConfigValidationError("dataset.ca_cert_path must be null or a non-empty path")
    if cfg.s3 is not None:
        if not cfg.s3.endpoint_url.strip():
            raise ConfigValidationError("s3.endpoint_url must not be empty")
        if not cfg.s3.dataset_bucket.strip():
            raise ConfigValidationError("s3.dataset_bucket must not be empty")
        if not cfg.s3.artifacts_bucket.strip():
            raise ConfigValidationError("s3.artifacts_bucket must not be empty")
    if cfg.mlflow.enabled and cfg.mlflow.tracking_uri is not None and not cfg.mlflow.tracking_uri.strip():
        raise ConfigValidationError("mlflow.tracking_uri must be null or a non-empty URI")
    if cfg.optuna.enabled:
        if cfg.optuna.n_trials <= 0:
            raise ConfigValidationError("optuna.n_trials must be > 0")
        if cfg.optuna.top_k <= 0:
            raise ConfigValidationError("optuna.top_k must be > 0")
        if cfg.optuna.direction not in {"maximize", "minimize"}:
            raise ConfigValidationError("optuna.direction must be 'maximize' or 'minimize'")
        if cfg.optuna.lr_min <= 0 or cfg.optuna.lr_max <= 0 or cfg.optuna.lr_min > cfg.optuna.lr_max:
            raise ConfigValidationError("optuna lr bounds are invalid")
    if cfg.quality_gate.enabled:
        if cfg.quality_gate.min_val_psnr is not None and cfg.quality_gate.min_val_psnr < 0:
            raise ConfigValidationError("quality_gate.min_val_psnr must be >= 0")
        if cfg.quality_gate.min_val_ssim is not None:
            if not (0.0 <= cfg.quality_gate.min_val_ssim <= 1.0):
                raise ConfigValidationError("quality_gate.min_val_ssim must be in [0, 1]")
        if cfg.quality_gate.max_val_loss is not None and cfg.quality_gate.max_val_loss < 0:
            raise ConfigValidationError("quality_gate.max_val_loss must be >= 0")


def _expand_env_in_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env_in_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_in_obj(v) for v in value]
    if isinstance(value, str):
        # Supports ${VAR} / $VAR and ~ user home expansion.
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _validate_no_unexpanded_vars(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _validate_no_unexpanded_vars(v, f"{path}.{k}")
        return
    if hasattr(value, "__dataclass_fields__"):
        for field_name in value.__dataclass_fields__:
            _validate_no_unexpanded_vars(getattr(value, field_name), f"{path}.{field_name}")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _validate_no_unexpanded_vars(item, f"{path}[{idx}]")
        return
    if isinstance(value, str) and ("${" in value or "$" in value):
        raise ConfigValidationError(
            f"Unexpanded environment variable in {path}: {value}. "
            "Set the variable in your shell/CI before running."
        )
