import math
from src.metrics import calculate_metrics

def test_metrics_known_values():
    m=calculate_metrics([0,0,1,1],[0,1,0,1],[.1,.8,.3,.9])
    assert m["tn"]==1 and m["fp"]==1 and m["fn"]==1 and m["tp"]==1
    assert m["accuracy"]==.5 and m["specificity"]==.5 and m["balanced_accuracy"]==.5
    assert m["f1"]==.5 and m["roc_auc"]==.75

def test_zero_division_and_auc_missing():
    m=calculate_metrics([0,0],[0,0],[.1,.2])
    assert m["precision"]==0 and m["recall"]==0 and math.isnan(m["roc_auc"])
