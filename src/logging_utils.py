"""
Logging utilities for SPA-VAEGAN training.
Writes real-time metrics to CSV files and TensorBoard summaries.
"""

import csv
import os
import time
from typing import Any, Dict, List, Optional


class CSVLogger:
    """Appends training and validation logs to a CSV file."""
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write header if file does not exist or is empty
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row_dict: Dict[str, Any]) -> None:
        """Append a single row of metrics."""
        filtered_row = {k: row_dict.get(k, "") for k in self.fieldnames}
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(filtered_row)


class TrainingMonitor:
    """Handles iteration timing, CSV logging, and console status formatting."""
    def __init__(self, csv_path: str, fieldnames: List[str], use_tensorboard: bool = True):
        self.logger = CSVLogger(csv_path, fieldnames)
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(os.path.dirname(csv_path), "tb")
                self.writer = SummaryWriter(log_dir=tb_dir)
            except Exception:
                self.writer = None
        self.start_time = time.time()
        self.it_start = time.time()

    def tic(self):
        self.it_start = time.time()

    def log_step(self, iteration: int, metrics: Dict[str, Any], print_console: bool = False):
        step_time = time.time() - self.it_start
        metrics["iteration"] = iteration
        metrics["step_time"] = round(step_time, 4)
        
        self.logger.log(metrics)
        
        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, iteration)

        if print_console:
            log_str = f"Iter [{iteration:05d}] ({step_time*1000:.1f}ms) | "
            log_str += " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items() if k not in ("iteration", "step_time"))
            print(log_str)

    def close(self):
        if self.writer is not None:
            self.writer.close()
