import pandas as pd
from src.config import ExperimentConfig
from src.experiment import run_experiment

def test_three_strategies_shared_test_and_shape():
    cfg=ExperimentConfig(drift_levels=2,repeats=1,n_train=80,n_adapt=60,n_test=100,seed=3)
    r=run_experiment(cfg)
    assert set(r.detail.strategy)=={"baseline","recent","combined"}
    assert len(r.detail)==6
    for _,g in r.detail.groupby(["repeat","intensity"]):
        assert g.test_seed.nunique()==1
    assert (r.detail.n_test==100).all()
    assert r.aggregate.shape[0]==6

def test_reproducible_metrics():
    cfg=ExperimentConfig(drift_levels=2,repeats=1,n_train=80,n_adapt=60,n_test=100,seed=3)
    a=run_experiment(cfg); b=run_experiment(cfg)
    assert a.detail["f1"].tolist()==b.detail["f1"].tolist()

def test_validation():
    try: ExperimentConfig(n_train=0).validate()
    except ValueError: return
    assert False


def test_concept_drift_hurts_baseline_and_recent_helps():
    cfg=ExperimentConfig(drift_type="concept",drift_levels=2,repeats=3,n_train=300,n_adapt=300,n_test=1000,seed=5)
    a=run_experiment(cfg).aggregate
    at1=a[a.intensity==1.0].set_index("strategy")["accuracy_mean"]
    assert at1["recent"]>at1["baseline"]+0.05

def test_covariate_shift_hurts_baseline():
    cfg=ExperimentConfig(drift_type="covariate",drift_levels=2,repeats=3,n_train=300,n_adapt=300,n_test=1000,seed=5)
    a=run_experiment(cfg).aggregate
    b=a[a.strategy=="baseline"].set_index("intensity")["accuracy_mean"]
    assert b[0.0]>b[1.0]+0.1

def test_oracle_is_upper_bound():
    cfg=ExperimentConfig(drift_levels=2,repeats=2,n_train=300,n_adapt=300,n_test=2000,seed=9)
    a=run_experiment(cfg).aggregate
    assert (a.accuracy_mean<=a.oracle_accuracy_mean+0.02).all()

def test_ci_uses_student_t():
    import pandas as pd
    from src.statistics import aggregate_results
    df=pd.DataFrame({"drift_type":"c","intensity":0.0,"strategy":"baseline","model":"m","accuracy":[0.8,0.82,0.78,0.81,0.79]})
    row=aggregate_results(df).iloc[0]
    half=row["accuracy_ci95_high"]-row["accuracy_mean"]
    assert abs(half-2.776*row["accuracy_std"]/5**0.5)<1e-3

def test_paired_difference():
    from src.statistics import paired_difference
    r=run_experiment(ExperimentConfig(drift_levels=2,repeats=3,n_train=80,n_adapt=60,n_test=100,seed=1))
    p=paired_difference(r.detail,"recent","baseline")
    assert len(p)==2 and (p.n==3).all()
