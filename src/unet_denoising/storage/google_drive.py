from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from unet_denoising.exceptions import ConfigValidationError


@dataclass
class GoogleDriveStorage:
    """Local path wrapper for mounted Google Drive directories."""

    drive_root: Path
    experiment_name: str

    @property
    def experiment_dir(self) -> Path:
        return self.drive_root / "experiments" / self.experiment_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.experiment_dir / "checkpoints"

    @property
    def outputs_dir(self) -> Path:
        return self.experiment_dir / "outputs"

    @property
    def plots_dir(self) -> Path:
        return self.experiment_dir / "plots"

    def ensure_dirs(self) -> None:
        try:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            self.plots_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigValidationError(
                "Failed to initialize storage directories under "
                f"'{self.drive_root}'. Set 'storage.google_drive_root' in config "
                "to a writable local path (or mounted drive path) for this machine."
            ) from exc

    @property
    def runs_dir(self) -> Path:
        return self.experiment_dir / "runs"

    def create_run_dir(self, run_type: str, run_id: str | None = None) -> Path:
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / f"{run_type}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
