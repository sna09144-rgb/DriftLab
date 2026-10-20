"""Reproduce the article's Table 1 and paired tests. Run from project root."""
import pandas as pd
from src.config import article_config
from src.experiment import run_experiment
from src.statistics import paired_comparisons

frames=[]
for drift_type in ("covariate", "concept"):
    result=run_experiment(article_config(drift_type))
    result.aggregate.to_csv(f"table1_{drift_type}_aggregate.csv", index=False)
    result.detail.to_csv(f"table1_{drift_type}_detail.csv", index=False)
    paired_comparisons(result.detail).to_csv(f"table1_{drift_type}_paired.csv", index=False)
    top=result.aggregate[result.aggregate.intensity.eq(1.0)].copy()
    top.insert(0, "scenario", drift_type)
    frames.append(top)
pd.concat(frames, ignore_index=True).to_csv("table1_results.csv", index=False)
print("Saved table1_results.csv and per-scenario detail/aggregate/paired CSV files.")
