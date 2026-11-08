"""Append-only experiment log (model, hyperparameters, metrics) for reproducibility."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

LOG = Path(__file__).resolve().parents[1] / "reports" / "experiments.csv"


def log_experiment(task: str, model: str, params: dict, metrics: dict) -> None:
    """Append one experiment row to reports/experiments.csv."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "model": model,
        "params": json.dumps(params, default=str),
        "metrics": json.dumps({k: round(float(v), 4) for k, v in metrics.items()}),
    }
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)