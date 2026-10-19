import numpy as np
from src.data_generator import generate_dataset

def test_shapes_finite_labels():
    d=generate_dataset(100,"covariate",.5,12)
    assert d.X.shape==(100,2) and d.y.shape==(100,)
    assert np.isfinite(d.X).all() and set(np.unique(d.y)).issubset({0,1})

def test_reproducibility_and_seed():
    a=generate_dataset(100,"concept",.4,7)
    b=generate_dataset(100,"concept",.4,7)
    c=generate_dataset(100,"concept",.4,8)
    assert np.array_equal(a.X,b.X) and np.array_equal(a.y,b.y)
    assert not np.array_equal(a.X,c.X)

def test_zero_drift_same_distribution_transform():
    a=generate_dataset(300,"covariate",0,10)
    b=generate_dataset(300,"covariate",0,10)
    assert np.array_equal(a.X,b.X) and np.array_equal(a.y,b.y)

def test_covariate_shift_changes_features():
    a=generate_dataset(1000,"covariate",0,1)
    b=generate_dataset(1000,"covariate",1,1)
    assert abs(a.X[:,1].mean()-b.X[:,1].mean())>.8

def test_concept_drift_changes_labels_rule_statistically():
    a=generate_dataset(5000,"concept",0,20,0)
    b=generate_dataset(5000,"concept",1,20,0)
    assert np.mean(a.y != b.y) > .15


def test_bayes_oracle_beats_random_and_is_upper_bound_like():
    from src.data_generator import bayes_predict
    d=generate_dataset(20000,"covariate",1,3,0.05)
    acc=(bayes_predict(d.X,"covariate",1)==d.y).mean()
    assert acc>0.7
