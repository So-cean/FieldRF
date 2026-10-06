"""TensorBoard + CSV dual logger for training and validation metrics."""

from pathlib import Path
import csv


class TrainLogger:
    """Dual writer: TensorBoard scalars + CSVs.

    Args:
        log_dir: Directory to write logs into.
        resume: If True, CSV files are appended rather than overwritten.
        experiment: Experiment name (e.g. task_name).  Applied consistently
                    to both training and validation CSVs.
    """

    def __init__(self, log_dir: str | Path, resume: bool = False,
                 experiment: str = ""):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._resume = resume
        self._experiment = experiment

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(str(self._log_dir), purge_step=None if resume else 0)

        # Training CSV
        self._train_path = self._log_dir / "training_log.csv"
        self._train_header = ["experiment", "epoch", "step", "loss_D", "loss_G",
                              "lr_G", "lr_D", "lr"]
        if not (resume and self._train_path.exists()):
            self._write_header(self._train_path, self._train_header)

        # Validation CSV — header written on first log_val call
        self._val_path = self._log_dir / "val_metrics.csv"
        self._val_header = None
        if resume and self._val_path.exists():
            with open(self._val_path) as f:
                self._val_header = next(csv.reader(f))

    # ── Training logging ────────────────────────────────────────────

    def log_train(self, epoch: int, step: int,
                  loss_D: float, loss_G: float,
                  lr_G: float = 0.0, lr_D: float = 0.0,
                  lr: float = 0.0):
        """Log one epoch's training metrics.

        *lr_G* / *lr_D* for pretrain (G/D/F optimizers);
        *lr* for finetune (single optimizer).
        """
        self._writer.add_scalar("loss/D", loss_D, step)
        self._writer.add_scalar("loss/G", loss_G, step)
        if lr_G > 0:
            self._writer.add_scalar("lr/G", lr_G, step)
        if lr_D > 0:
            self._writer.add_scalar("lr/D", lr_D, step)
        if lr > 0:
            self._writer.add_scalar("lr/G", lr, step)
        self._append(self._train_path, [self._experiment, epoch, step,
                                         loss_D, loss_G, lr_G, lr_D, lr])

    def log_finetune(self, epoch: int, step: int, loss: float, lr: float = 0.0):
        """Log one finetune epoch's training metrics (L1+LPIPS+...)."""
        self._writer.add_scalar("train/loss", loss, step)
        if lr > 0:
            self._writer.add_scalar("lr/G", lr, step)
        self._append(self._train_path, [self._experiment, epoch, step,
                                         0.0, loss, 0.0, 0.0, lr])

    # ── Validation logging ──────────────────────────────────────────

    def log_val(self, epoch: int, **metrics: float):
        if self._val_header is None:
            self._val_header = ["experiment", "epoch"] + list(metrics.keys())
            if not (self._resume and self._val_path.exists()):
                self._write_header(self._val_path, self._val_header)
        for k, v in metrics.items():
            self._writer.add_scalar(f"val/{k}", v, epoch)
        self._append(self._val_path, [self._experiment, epoch] + list(metrics.values()))

    # ── Helpers ─────────────────────────────────────────────────────

    def close(self):
        self._writer.close()

    @staticmethod
    def _write_header(path, columns):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(columns)

    @staticmethod
    def _append(path, row):
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(row)
            f.flush()
