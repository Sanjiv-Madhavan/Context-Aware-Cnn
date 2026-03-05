from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from conftest import make_config
from unet_denoising.data import download as dl


class FakeResp(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_file_with_fake_opener(tmp_path, monkeypatch):
    class FakeOpener:
        def open(self, url):
            return FakeResp(b"abc")

    monkeypatch.setattr(dl.urllib.request, "build_opener", lambda *a, **k: FakeOpener())
    out = tmp_path / "x.bin"
    dl.download_file("http://x", out, verify_ssl=False)
    assert out.read_bytes() == b"abc"


def test_extract_and_organize(tmp_path):
    cfg = make_config(tmp_path, with_s3=False)
    extract_root = tmp_path / "extract"
    (extract_root / "training/patches_input").mkdir(parents=True)
    (extract_root / "training/patches_gt").mkdir(parents=True)
    (extract_root / "validation/large_input").mkdir(parents=True)
    (extract_root / "validation/large_gt").mkdir(parents=True)

    arr = np.ones((4, 4), dtype=np.float32)
    tifffile.imwrite(extract_root / "training/patches_input/a.tif", arr)
    tifffile.imwrite(extract_root / "training/patches_gt/a.tif", arr)
    tifffile.imwrite(extract_root / "validation/large_input/a.tif", arr)
    tifffile.imwrite(extract_root / "validation/large_gt/a.tif", arr)

    counts = dl.organize_dataset(extract_root, cfg.paths)
    assert counts["train_noisy"] == 1
    assert counts["train_gt"] == 1
    assert counts["val_noisy"] == 1
    assert counts["val_gt"] == 1


def test_extract_zip(tmp_path):
    z = tmp_path / "a.zip"
    dst = tmp_path / "out"
    with zipfile.ZipFile(z, "w") as zp:
        zp.writestr("f.txt", "hello")
    dl.extract_zip(z, dst)
    assert (dst / "f.txt").read_text() == "hello"
