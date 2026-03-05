from __future__ import annotations

from types import SimpleNamespace

import pytest

from unet_denoising.cli import _validate_env_for_command, build_parser
from unet_denoising.exceptions import ConfigValidationError


def test_parser_has_commands():
    parser = build_parser()
    help_text = parser.format_help()
    assert "download" in help_text
    assert "s3-push" in help_text
    assert "s3-pull" in help_text
    assert "train" in help_text
    assert "infer" in help_text
    assert "sweep-optuna" in help_text


def test_validate_env_for_command_s3(monkeypatch):
    cfg = SimpleNamespace(s3=SimpleNamespace(auto_pull=False))
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    with pytest.raises(ConfigValidationError):
        _validate_env_for_command("s3-push", cfg)


def test_validate_env_for_command_no_s3_needed():
    cfg = SimpleNamespace(s3=None)
    _validate_env_for_command("train", cfg)
