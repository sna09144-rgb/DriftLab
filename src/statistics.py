import math
import numpy as np
import pandas as pd
from scipy import stats

def aggregate_results(detail: pd.DataFrame, group_cols=None):
    if detail.empty: return pd.DataFrame()
    group_cols = group_cols or ["drift_type", "intensity", "strategy", "model"]
    metric_cols = [c for c in ["accuracy","precision","recall","f1","specificity","balanced_accuracy","roc_auc","oracle_accuracy","training_seconds"] if c in detail.columns]
    rows = []
    for keys, group in detail.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_successful_repeats"] = int(len(group))
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            n = len(vals)
            mean = float(np.mean(vals)) if n else float("nan")
            std = float(np.std(vals, ddof=1)) if n >= 2 else float("nan")
            # 95% доверительный интервал по распределению Стьюдента (df = n-1)
            half = float(stats.t.ppf(0.975, n - 1)) * std / math.sqrt(n) if n >= 2 else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(detail: pd.DataFrame, strategy_a: str, strategy_b: str, metric: str = "accuracy"):
    """Парная разность метрики (a - b) по повторам на одной тестовой выборке.
    Возвращает среднее, t-интервал 95% и признак «интервал не содержит 0» (описательно)."""
    rows = []
    for intensity, g in detail.groupby("intensity", sort=True):
        a = g[g.strategy == strategy_a].set_index("repeat")[metric]
        b = g[g.strategy == strategy_b].set_index("repeat")[metric]
        d = (a - b).dropna().to_numpy(float)
        n = len(d)
        mean = float(np.mean(d)) if n else float("nan")
        if n >= 2:
            half = float(stats.t.ppf(0.975, n - 1)) * float(np.std(d, ddof=1)) / math.sqrt(n)
        else:
            half = float("nan")
        rows.append({"intensity": float(intensity), "n": n, "diff_mean": mean,
                     "ci95_low": mean - half, "ci95_high": mean + half,
                     "excludes_zero": bool(n >= 2 and (mean - half > 0 or mean + half < 0))})
    return pd.DataFrame(rows)
