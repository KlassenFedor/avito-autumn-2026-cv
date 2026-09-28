"""Brier score для бинарной задачи и подбор температуры; только NumPy."""
import numpy as np


def sigmoid(logits):
    z = np.clip(np.asarray(logits, dtype=np.float64), -80, 80)
    return 1.0 / (1.0 + np.exp(-z))


def metrics(labels, probabilities):
    y, p = np.asarray(labels, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Нужны непустые одномерные массивы одинаковой длины")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Некорректные метки или вероятности")
    brier = float(np.mean((p - y) ** 2))
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {"n": len(y), "positive_rate": float(y.mean()), "brier": brier,
            "score": 1 - brier, "accuracy": float(np.mean((p >= 0.5) == y)),
            "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))}


def fit_temperature(logits, labels):
    # небольшая фиксированная сетка; T=1 в ней есть, поэтому калибровка не может ухудшить loss
    candidates = np.unique(np.r_[1.0, np.geomspace(0.25, 8.0, 161)])
    losses = [metrics(labels, sigmoid(np.asarray(logits) / t))["brier"] for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def grouped_metrics(rows, probabilities):
    y = np.asarray([r["label"] for r in rows])
    p = np.asarray(probabilities)
    result = {"all": metrics(y, p)}
    for source in sorted({r["source"] for r in rows}):
        mask = np.asarray([r["source"] == source for r in rows])
        result[f"source={source}"] = metrics(y[mask], p[mask])
    return result


def probability_temperature(probabilities, temperature, eps=1e-6):
    """Температура по логитам, восстановленным из вероятностей; T=1 входит в сетку точно."""
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Некорректные вероятности")
    if not np.isfinite(temperature) or temperature <= 0 or not 0 < eps < 0.5:
        raise ValueError("Некорректные temperature или epsilon")
    if temperature == 1.0:
        return p.copy()
    clipped = np.clip(p, eps, 1 - eps)
    return sigmoid((np.log(clipped) - np.log1p(-clipped)) / temperature)


def fit_probability_temperature(probabilities, labels, eps=1e-6):
    candidates = np.unique(np.r_[1.0, np.geomspace(0.25, 8.0, 161)])
    losses = [metrics(labels, probability_temperature(probabilities, t, eps))["brier"] for t in candidates]
    return float(candidates[int(np.argmin(losses))])
