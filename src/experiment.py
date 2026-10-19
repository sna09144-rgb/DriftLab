from dataclasses import dataclass
from datetime import datetime, timezone
import time, uuid
import numpy as np
import pandas as pd
from .config import ExperimentConfig, STRATEGIES
from .data_generator import generate_dataset, generate_reference, bayes_predict
from .models import fit_model, predict_model, positive_scores
from .metrics import calculate_metrics
from .statistics import aggregate_results

@dataclass
class ExperimentResult:
    experiment_id: str
    created_at: str
    config: dict
    detail: pd.DataFrame
    aggregate: pd.DataFrame
    examples: dict

def run_experiment(config: ExperimentConfig, progress_callback=None) -> ExperimentResult:
    config.validate()
    exp_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat()
    intensities = np.linspace(config.drift_min, config.drift_max, config.drift_levels)
    rows, examples = [], {}
    total = config.repeats * len(intensities)
    step = 0
    for repeat in range(config.repeats):
        # Deterministic, separated seed streams per repeat and purpose.
        seed_seq = np.random.SeedSequence([config.seed, repeat, 731])
        seeds = seed_seq.generate_state(4, dtype=np.uint32)
        base = generate_reference(config.n_train, int(seeds[0]), config.label_noise)
        baseline = fit_model(base.X, base.y, config.model, int(seeds[1]))
        for level_idx, intensity in enumerate(intensities):
            # Adaptation data and test data have independent, purpose-specific seeds.
            adapt_seed = int(np.random.SeedSequence([config.seed, repeat, level_idx, 991]).generate_state(1)[0])
            test_seed = int(np.random.SeedSequence([config.seed, repeat, level_idx, 1777]).generate_state(1)[0])
            adapt = generate_dataset(config.n_adapt, config.drift_type, float(intensity), adapt_seed, config.label_noise)
            test = generate_dataset(config.n_test, config.drift_type, float(intensity), test_seed, config.label_noise)
            if level_idx == len(intensities)//2 and repeat == 0:
                examples = {"X": test.X, "y": test.y, "intensity": float(intensity),
                            "baseline_model": baseline, "test_seed": test_seed,
                            "reference_X": base.X, "reference_y": base.y}
            oracle_acc = float(np.mean(bayes_predict(test.X, config.drift_type, float(intensity)) == test.y))
            for strategy in config.strategies:
                if strategy == "baseline":
                    model = baseline
                    fit_seconds = float("nan")  # trained once before the drift loop
                    n_old, n_new, n_used = config.n_train, 0, config.n_train
                elif strategy == "recent":
                    start = time.perf_counter()
                    model = fit_model(adapt.X, adapt.y, config.model, int(seeds[2]))
                    fit_seconds = float(time.perf_counter()-start)
                    n_old, n_new, n_used = 0, config.n_adapt, config.n_adapt
                elif strategy == "combined":
                    X_train = np.concatenate([base.X, adapt.X])
                    y_train = np.concatenate([base.y, adapt.y])
                    start = time.perf_counter()
                    model = fit_model(X_train, y_train, config.model, int(seeds[3]))
                    fit_seconds = float(time.perf_counter()-start)
                    n_old, n_new, n_used = config.n_train, config.n_adapt, config.n_train+config.n_adapt
                else: continue
                pred = predict_model(model, test.X)
                score = positive_scores(model, test.X)
                metrics = calculate_metrics(test.y, pred, score)
                rows.append({
                    "experiment_id": exp_id, "created_at": created, "drift_type": config.drift_type,
                    "intensity": float(intensity), "repeat": repeat, "seed": int(config.seed),
                    "adapt_seed": adapt_seed, "test_seed": test_seed, "strategy": strategy,
                    "strategy_name": STRATEGIES[strategy], "model": config.model,
                    "n_old_train": n_old, "n_new_train": n_new, "n_train_total": n_used,
                    "n_test": config.n_test, "training_seconds": fit_seconds,
                    "oracle_accuracy": oracle_acc, **metrics
                })
            step += 1
            if progress_callback: progress_callback(step / total)
    detail = pd.DataFrame(rows)
    agg = aggregate_results(detail)
    cfg = config.to_dict()
    return ExperimentResult(exp_id, created, cfg, detail, agg, examples)
