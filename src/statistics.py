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


def paired_comparisons(detail: pd.DataFrame, metrics=("f1", "accuracy"),
                       strategy_pairs=(("recent", "baseline"), ("combined", "baseline"), ("recent", "combined"))):
    """Paired t tests per drift intensity, with t-based 95% CIs and Holm correction.

    Holm correction is applied separately for each metric across all requested
    strategy-pair × intensity comparisons. Pairing is by repeat; the experiment
    evaluates strategies on the same test set within each repeat/intensity.
    """
    from scipy import stats
    rows = []
    for metric in metrics:
        metric_rows = []
        for intensity, group in detail.groupby("intensity", sort=True):
            wide = group.pivot(index="repeat", columns="strategy", values=metric)
            for a, b in strategy_pairs:
                if a not in wide or b not in wide:
                    continue
                d = (wide[a] - wide[b]).dropna().to_numpy(float)
                n = len(d)
                mean = float(np.mean(d)) if n else float("nan")
                sd = float(np.std(d, ddof=1)) if n >= 2 else float("nan")
                half = float(stats.t.ppf(0.975, n-1) * sd / math.sqrt(n)) if n >= 2 else float("nan")
                p = float(stats.ttest_1samp(d, 0.0).pvalue) if n >= 2 else float("nan")
                metric_rows.append({"metric": metric, "intensity": float(intensity),
                    "contrast": f"{a}-{b}", "n": n, "mean_diff": mean, "sd_diff": sd,
                    "ci95_low": mean-half, "ci95_high": mean+half, "p_raw": p})
        valid = [i for i, r in enumerate(metric_rows) if np.isfinite(r["p_raw"])]
        if valid:
            # Holm step-down adjustment without an additional dependency.
            ordered = sorted(valid, key=lambda i: metric_rows[i]["p_raw"])
            m = len(ordered)
            running_max = 0.0
            adjusted = {}
            for rank, idx in enumerate(ordered):
                candidate = min(1.0, (m - rank) * metric_rows[idx]["p_raw"])
                running_max = max(running_max, candidate)
                adjusted[idx] = running_max
            for i in valid:
                p_adj = adjusted[i]
                metric_rows[i]["p_holm"] = float(p_adj)
                metric_rows[i]["reject_holm_0_05"] = bool(p_adj < 0.05)
        for r in metric_rows:
            r.setdefault("p_holm", float("nan")); r.setdefault("reject_holm_0_05", False)
        rows.extend(metric_rows)
    return pd.DataFrame(rows)
