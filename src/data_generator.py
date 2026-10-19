from dataclasses import dataclass
import numpy as np

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    drift_type: str
    intensity: float
    seed: int

CURVATURE = 1.5      # нелинейность истинного правила (линейная модель его не описывает точно)
COV_SHIFT = 1.5      # смещение второго признака при d=1 (covariate shift)


def _logits(X, angle=0.0):
    """Истинные логиты правила: 2.4*(X·w) + CURVATURE*(x1^2 - x2^2)."""
    theta = np.pi / 4 + angle
    w = np.array([np.cos(theta), np.sin(theta)])
    return 2.4 * (X @ w) + CURVATURE * (X[:, 0] ** 2 - X[:, 1] ** 2)


def bayes_predict(X, drift_type: str, intensity: float):
    """Оптимальный (oracle) классификатор: знает истинное правило.
    Шум меток < 0.5 не меняет знак оптимального решения."""
    angle = intensity * (np.pi / 2.0) if drift_type == "concept" else 0.0
    return (_logits(np.asarray(X), angle) > 0).astype(int)


def _labels(X, angle=0.0, noise=0.0, rng=None):
    # Общее условное правило: P(Y=1|X)=sigmoid(_logits).
    # При covariate shift angle фиксирован, меняется только P(X).
    logits = _logits(X, angle)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    y = (rng.random(len(X)) < probs).astype(int)
    if noise:
        flip = rng.random(len(X)) < noise
        y[flip] = 1 - y[flip]
    return y

def generate_dataset(n: int, drift_type: str = "covariate", intensity: float = 0.0,
                     seed: int = 42, label_noise: float = 0.05) -> Dataset:
    if n < 1: raise ValueError("n должен быть положительным")
    if drift_type not in {"covariate", "concept"}: raise ValueError("Неизвестный тип дрейфа")
    if not 0 <= intensity <= 1: raise ValueError("intensity должен быть в [0, 1]")
    if not 0 <= label_noise < .5: raise ValueError("label_noise должен быть в [0, .5)")
    rng = np.random.default_rng(seed)
    # Two-class-like feature mixture, generated independently of labels.
    y_latent = rng.integers(0, 2, size=n)
    means = np.where(y_latent[:, None] == 1, np.array([1.0, 0.8]), np.array([-1.0, -0.8]))
    X = rng.normal(size=(n, 2)) * 0.85 + means
    # Covariate shift transforms X and then labels are sampled from the same
    # fixed conditional mechanism; concept drift changes only the mechanism.
    if drift_type == "covariate":
        X = X + np.array([0.0, COV_SHIFT * intensity])
        angle = 0.0
    else:
        angle = intensity * (np.pi / 2.0)
    y = _labels(X, angle=angle, noise=label_noise, rng=rng)
    return Dataset(X=X.astype(float), y=y.astype(int), drift_type=drift_type,
                   intensity=float(intensity), seed=int(seed))

def generate_reference(n: int, seed: int, label_noise: float = 0.05) -> Dataset:
    return generate_dataset(n, "covariate", 0.0, seed, label_noise)
