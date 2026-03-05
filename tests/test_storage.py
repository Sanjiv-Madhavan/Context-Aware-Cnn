from pathlib import Path

from unet_denoising.storage.google_drive import GoogleDriveStorage


def test_storage_run_dir_creation(tmp_path):
    storage = GoogleDriveStorage(drive_root=Path(tmp_path), experiment_name="exp")
    storage.ensure_dirs()
    run_dir = storage.create_run_dir("train", run_id="20260304_120000")
    assert run_dir.exists()
    assert run_dir.name == "train_20260304_120000"
    assert storage.checkpoints_dir.exists()
