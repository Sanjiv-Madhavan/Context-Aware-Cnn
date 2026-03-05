import pytest

from unet_denoising.exceptions import ConfigValidationError
from unet_denoising.validation.runtime import ensure_dir_exists, ensure_env_vars, ensure_file_exists, ensure_writable_dir


def test_validation_helpers(tmp_path):
    d = tmp_path / "d"
    ensure_writable_dir(d, "d")
    assert d.exists()

    f = d / "x.txt"
    f.write_text("ok")

    assert ensure_dir_exists(d, "d") == d
    assert ensure_file_exists(f, "f") == f


@pytest.mark.parametrize(
    "fn,arg,label",
    [
        (ensure_dir_exists, "missing", "missing dir"),
        (ensure_file_exists, "missing.txt", "missing file"),
    ],
)
def test_validation_helpers_fail(tmp_path, fn, arg, label):
    with pytest.raises(ConfigValidationError):
        fn(tmp_path / arg, label)


def test_ensure_env_vars(monkeypatch):
    monkeypatch.setenv("A", "1")
    monkeypatch.setenv("B", "2")
    ensure_env_vars(["A", "B"], "test")


def test_ensure_env_vars_fail(monkeypatch):
    monkeypatch.delenv("A", raising=False)
    with pytest.raises(ConfigValidationError):
        ensure_env_vars(["A"], "test")
