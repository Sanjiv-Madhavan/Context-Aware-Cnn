from __future__ import annotations

from pathlib import Path

from conftest import make_config, seed_tif_tree
from unet_denoising.pipelines import s3_pipeline


class FakePaginator:
    def __init__(self, objects):
        self._objects = objects

    def paginate(self, Bucket, Prefix):
        contents = [{"Key": k} for k in sorted(self._objects) if k.startswith(Prefix)]
        return [{"Contents": contents}]


class FakeS3Client:
    def __init__(self):
        self.buckets = set()
        self.objects = {}

    def list_buckets(self):
        return {"Buckets": [{"Name": b} for b in sorted(self.buckets)]}

    def create_bucket(self, Bucket, **kwargs):
        self.buckets.add(Bucket)

    def upload_file(self, src, bucket, key):
        self.buckets.add(bucket)
        self.objects[(bucket, key)] = Path(src).read_bytes()

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        all_keys = [k for (b, k) in self.objects.keys()]
        return FakePaginator(all_keys)

    def download_file(self, bucket, key, out_path):
        data = self.objects[(bucket, key)]
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def test_s3_push_pull(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, with_s3=True)
    seed_tif_tree(cfg)

    exp_dir = Path(cfg.storage.root_dir) / "experiments/exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "normalization_stats.json").write_text('{"noisy_mean":0,"noisy_std":1}')

    fake = FakeS3Client()
    monkeypatch.setattr(s3_pipeline, "_build_client", lambda cfg_: fake)

    s3_pipeline.run_s3_push(cfg)
    assert fake.objects

    # clear local data then pull back
    for p in Path(cfg.paths.noisy_train_dir).glob("*.tif"):
        p.unlink()

    s3_pipeline.run_s3_pull(cfg)
    assert list(Path(cfg.paths.noisy_train_dir).glob("*.tif"))
