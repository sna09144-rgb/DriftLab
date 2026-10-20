import pytest
from src.config import article_config
from src.experiment import run_experiment

@pytest.mark.parametrize("drift_type,expected", [
    ("covariate", {"baseline": (0.443433, 0.554928), "recent": (0.650833, 0.629287), "combined": (0.499567, 0.511873)}),
    ("concept", {"baseline": (0.498233, 0.530438), "recent": (0.597467, 0.655491), "combined": (0.546333, 0.587578)}),
])
def test_article_table1_reproduces_within_tolerance(drift_type, expected):
    result = run_experiment(article_config(drift_type))
    table = result.aggregate[result.aggregate.intensity.eq(1.0)].set_index("strategy")
    for strategy, (accuracy, f1) in expected.items():
        assert abs(table.loc[strategy, "accuracy_mean"] - accuracy) <= 0.001
        assert abs(table.loc[strategy, "f1_mean"] - f1) <= 0.001
