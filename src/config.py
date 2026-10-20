from dataclasses import dataclass, asdict
from typing import Literal

DRIFT_TYPES = {"covariate": "Covariate shift", "concept": "Concept drift"}
STRATEGIES = {"baseline": "Исходная модель", "recent": "Переобучение на новых данных", "combined": "Объединение данных"}
METRICS = ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy", "roc_auc"]
METRIC_LABELS = {"accuracy":"Accuracy", "precision":"Precision", "recall":"Recall", "f1":"F1-score", "specificity":"Specificity", "balanced_accuracy":"Balanced accuracy", "roc_auc":"ROC-AUC"}
FORMAT_VERSION = 1

@dataclass(frozen=True)
class ExperimentConfig:
    drift_type: Literal["covariate", "concept"] = "covariate"
    drift_min: float = 0.0
    drift_max: float = 1.0
    drift_levels: int = 5
    repeats: int = 5
    n_train: int = 500
    n_adapt: int = 300
    n_test: int = 1000
    label_noise: float = 0.05
    seed: int = 42
    model: str = "logistic_regression"
    strategies: tuple[str, ...] = ("baseline", "recent", "combined")

    def validate(self):
        if self.drift_type not in DRIFT_TYPES: raise ValueError("Неизвестный тип дрейфа")
        if not (0 <= self.drift_min <= self.drift_max <= 1): raise ValueError("Интенсивность должна находиться в диапазоне [0, 1]")
        if not 2 <= self.drift_levels <= 21: raise ValueError("Количество уровней должно быть от 2 до 21")
        if not 1 <= self.repeats <= 100: raise ValueError("Количество повторов должно быть от 1 до 100")
        if min(self.n_train, self.n_adapt, self.n_test) < 20: raise ValueError("Размер каждой выборки должен быть не меньше 20")
        if not 0 <= self.label_noise < 0.5: raise ValueError("Шум меток должен быть в диапазоне [0, 0.5)")
        if self.model != "logistic_regression": raise ValueError("Поддерживается logistic_regression")
        if not self.strategies or any(s not in STRATEGIES for s in self.strategies): raise ValueError("Выберите допустимые стратегии")
        return self

    def to_dict(self):
        result = asdict(self)
        result["strategies"] = list(self.strategies)
        return result


# Exact configuration used for the conference article and Table 1.
ARTICLE_PRESET = {
    "drift_min": 0.0, "drift_max": 1.0, "drift_levels": 6,
    "repeats": 30, "n_train": 500, "n_adapt": 300, "n_test": 1000,
    "label_noise": 0.05, "seed": 20261910,
}

def article_config(drift_type: str) -> ExperimentConfig:
    """Return the fixed, documented configuration used in the article."""
    if drift_type not in DRIFT_TYPES:
        raise ValueError("drift_type must be 'covariate' or 'concept'")
    return ExperimentConfig(drift_type=drift_type, **ARTICLE_PRESET).validate()
