import textwrap

import pytest

from unet_denoising.config import load_config
from unet_denoising.exceptions import ConfigValidationError


def test_load_config_ok(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              noisy_train_dir: "/tmp/noisy_train"
              gt_train_dir: "/tmp/gt_train"
              noisy_val_dir: "/tmp/noisy_val"
              gt_val_dir: "/tmp/gt_val"
              noisy_infer_dir: "/tmp/noisy_infer"
              gt_infer_dir: "/tmp/gt_infer"
            storage:
              google_drive_root: "/tmp/artifacts"
              experiment_name: "exp1"
            data:
              patch_size: 256
              border_size: 32
              crops_per_image: 10
              val_ratio: 0.2
              num_workers: 0
            train:
              seed: 1
              epochs: 5
              batch_size: 2
              lr: 0.001
              checkpoint_every: 1
            inference:
              checkpoint_name: "x.pth"
            """
        )
    )

    cfg = load_config(cfg_file)
    assert cfg.storage.experiment_name == "exp1"
    assert cfg.dataset is None


def test_load_config_invalid_val_ratio(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              noisy_train_dir: "/tmp/noisy_train"
              gt_train_dir: "/tmp/gt_train"
              noisy_val_dir: "/tmp/noisy_val"
              gt_val_dir: "/tmp/gt_val"
              noisy_infer_dir: "/tmp/noisy_infer"
              gt_infer_dir: "/tmp/gt_infer"
            storage:
              google_drive_root: "/tmp/artifacts"
              experiment_name: "exp1"
            data:
              patch_size: 256
              border_size: 32
              crops_per_image: 10
              val_ratio: 1.5
              num_workers: 0
            train:
              seed: 1
              epochs: 5
              batch_size: 2
              lr: 0.001
              checkpoint_every: 1
            inference:
              checkpoint_name: "x.pth"
            """
        )
    )

    with pytest.raises(ConfigValidationError):
        load_config(cfg_file)


def test_load_config_invalid_dataset(tmp_path):
    cfg_file = tmp_path / "bad_dataset.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              noisy_train_dir: "/tmp/noisy_train"
              gt_train_dir: "/tmp/gt_train"
              noisy_val_dir: "/tmp/noisy_val"
              gt_val_dir: "/tmp/gt_val"
              noisy_infer_dir: "/tmp/noisy_infer"
              gt_infer_dir: "/tmp/gt_infer"
            storage:
              google_drive_root: "/tmp/artifacts"
              experiment_name: "exp1"
            dataset:
              download_url: ""
              zip_path: "/tmp/a.zip"
              extract_dir: "/tmp/extract"
              overwrite: false
            data:
              patch_size: 256
              border_size: 32
              crops_per_image: 10
              val_ratio: 0.2
              num_workers: 0
            train:
              seed: 1
              epochs: 5
              batch_size: 2
              lr: 0.001
              checkpoint_every: 1
            inference:
              checkpoint_name: "x.pth"
            """
        )
    )

    with pytest.raises(ConfigValidationError):
        load_config(cfg_file)


def test_load_config_invalid_s3(tmp_path):
    cfg_file = tmp_path / "bad_s3.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              noisy_train_dir: "/tmp/noisy_train"
              gt_train_dir: "/tmp/gt_train"
              noisy_val_dir: "/tmp/noisy_val"
              gt_val_dir: "/tmp/gt_val"
              noisy_infer_dir: "/tmp/noisy_infer"
              gt_infer_dir: "/tmp/gt_infer"
            storage:
              google_drive_root: "/tmp/artifacts"
              experiment_name: "exp1"
            s3:
              endpoint_url: ""
              dataset_bucket: "data-bucket"
              artifacts_bucket: "art-bucket"
            data:
              patch_size: 256
              border_size: 32
              crops_per_image: 10
              val_ratio: 0.2
              num_workers: 0
            train:
              seed: 1
              epochs: 5
              batch_size: 2
              lr: 0.001
              checkpoint_every: 1
            inference:
              checkpoint_name: "x.pth"
            """
        )
    )

    with pytest.raises(ConfigValidationError):
        load_config(cfg_file)


def test_load_config_env_interpolation(tmp_path, monkeypatch):
    root = tmp_path / "project"
    monkeypatch.setenv("PROJECT_ROOT", str(root))

    cfg_file = tmp_path / "env.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              noisy_train_dir: "${PROJECT_ROOT}/data/small_images/noisy"
              gt_train_dir: "${PROJECT_ROOT}/data/small_images/gt"
              noisy_val_dir: "${PROJECT_ROOT}/data/large_images/noisy"
              gt_val_dir: "${PROJECT_ROOT}/data/large_images/gt"
              noisy_infer_dir: "${PROJECT_ROOT}/data/test_images/noisy"
              gt_infer_dir: "${PROJECT_ROOT}/data/test_images/gt"
            storage:
              root_dir: "${PROJECT_ROOT}/artifacts"
              experiment_name: "exp1"
            data:
              patch_size: 256
              border_size: 32
              crops_per_image: 10
              val_ratio: 0.2
              num_workers: 0
            train:
              seed: 1
              epochs: 5
              batch_size: 2
              lr: 0.001
              checkpoint_every: 1
            inference:
              checkpoint_name: "x.pth"
            """
        )
    )

    cfg = load_config(cfg_file)
    assert cfg.paths.noisy_train_dir == f"{root}/data/small_images/noisy"
    assert cfg.storage.root_dir == f"{root}/artifacts"


def test_load_config_unexpanded_env_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    cfg_file = tmp_path / "env_missing.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              noisy_train_dir: "${PROJECT_ROOT}/data/small_images/noisy"
              gt_train_dir: "${PROJECT_ROOT}/data/small_images/gt"
              noisy_val_dir: "${PROJECT_ROOT}/data/large_images/noisy"
              gt_val_dir: "${PROJECT_ROOT}/data/large_images/gt"
              noisy_infer_dir: "${PROJECT_ROOT}/data/test_images/noisy"
              gt_infer_dir: "${PROJECT_ROOT}/data/test_images/gt"
            storage:
              root_dir: "${PROJECT_ROOT}/artifacts"
              experiment_name: "exp1"
            data:
              patch_size: 256
              border_size: 32
              crops_per_image: 10
              val_ratio: 0.2
              num_workers: 0
            train:
              seed: 1
              epochs: 5
              batch_size: 2
              lr: 0.001
              checkpoint_every: 1
            inference:
              checkpoint_name: "x.pth"
            """
        )
    )

    with pytest.raises(ConfigValidationError):
        load_config(cfg_file)


def test_load_config_optuna_mlflow(tmp_path):
    cfg_file = tmp_path / "sweep.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            paths:
              noisy_train_dir: "/tmp/noisy_train"
              gt_train_dir: "/tmp/gt_train"
              noisy_val_dir: "/tmp/noisy_val"
              gt_val_dir: "/tmp/gt_val"
              noisy_infer_dir: "/tmp/noisy_infer"
              gt_infer_dir: "/tmp/gt_infer"
            storage:
              root_dir: "/tmp/artifacts"
              experiment_name: "exp1"
            mlflow:
              enabled: true
              tracking_uri: "file:/tmp/mlruns"
              experiment_name: "exp"
              run_name_prefix: "trial"
            optuna:
              enabled: true
              n_trials: 2
              top_k: 1
              study_name: "sweep"
              direction: "maximize"
              metric_name: "val_ssim"
              lr_min: 0.0001
              lr_max: 0.001
              batch_size_choices: [1]
              patch_size_choices: [4]
              border_size_choices: [1]
              crops_per_image_choices: [1]
              epochs_choices: [1]
            data:
              patch_size: 256
              border_size: 32
              crops_per_image: 10
              val_ratio: 0.2
              num_workers: 0
            train:
              seed: 1
              epochs: 5
              batch_size: 2
              lr: 0.001
              checkpoint_every: 1
            inference:
              checkpoint_name: "x.pth"
            """
        )
    )

    cfg = load_config(cfg_file)
    assert cfg.mlflow.enabled
    assert cfg.optuna.enabled
    assert cfg.optuna.n_trials == 2
