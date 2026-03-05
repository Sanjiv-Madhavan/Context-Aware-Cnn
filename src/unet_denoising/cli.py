from __future__ import annotations

import argparse
import sys

from unet_denoising.exceptions import DenoisingError
from unet_denoising.config import load_config
from unet_denoising.validation.runtime import ensure_env_vars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UNet denoising pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download and organize dataset")
    sub.add_parser("s3-push", help="Upload dataset/artifacts to S3 (LocalStack supported)")
    sub.add_parser("s3-pull", help="Download dataset/artifacts from S3 (LocalStack supported)")
    sub.add_parser("train", help="Run training pipeline")
    sub.add_parser("infer", help="Run inference pipeline")
    sub.add_parser("sweep-optuna", help="Run Optuna hyperparameter sweep and export top-K to MLflow")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    _validate_env_for_command(args.command, cfg)

    if args.command == "train":
        from unet_denoising.pipelines.train_pipeline import run_train

        run_train(cfg)
    elif args.command == "infer":
        from unet_denoising.pipelines.infer_pipeline import run_infer

        run_infer(cfg)
    elif args.command == "download":
        from unet_denoising.pipelines.download_pipeline import run_download

        run_download(cfg)
    elif args.command == "s3-push":
        from unet_denoising.pipelines.s3_pipeline import run_s3_push

        run_s3_push(cfg)
    elif args.command == "s3-pull":
        from unet_denoising.pipelines.s3_pipeline import run_s3_pull

        run_s3_pull(cfg)
    elif args.command == "sweep-optuna":
        from unet_denoising.pipelines.optuna_pipeline import run_optuna_sweep

        run_optuna_sweep(cfg)


def _validate_env_for_command(command: str, cfg) -> None:
    needs_s3_env = command in {"s3-push", "s3-pull"} or (command in {"train", "infer"} and cfg.s3 is not None and cfg.s3.auto_pull)
    if needs_s3_env:
        ensure_env_vars(
            ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
            context=f"command '{command}'",
        )


if __name__ == "__main__":
    try:
        main()
    except DenoisingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
