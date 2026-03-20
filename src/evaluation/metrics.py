import numpy as np

affinity_treshold = 7.0
solubility_treshold = 1.0


def compute_metrics(aff, sol, y_true=None, y_pred=None):
    metrics = {}
    metrics["best_affinity"] = float(np.max(aff))

    # Hit rate
    valid = (aff > affinity_treshold) & (sol > solubility_treshold)
    metrics["hit_rate"] = float(np.mean(valid))

    # Top-K
    k = min(10, len(aff))
    top_idx = np.argsort(aff)[-k:]
    metrics["top10_valid_rate"] = float(np.mean(valid[top_idx]))

    # Modelo
    if y_true is not None and y_pred is not None:
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics["rmse"] = float(rmse)
    else:
        metrics["rmse"] = None

    return metrics