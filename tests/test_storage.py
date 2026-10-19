from pathlib import Path
from src.config import ExperimentConfig
from src.experiment import run_experiment
from src.storage import save_result, load_result, list_experiments
from src.reporting import export_json, markdown_report
import json

def test_storage_roundtrip_and_exports(tmp_path):
    path=tmp_path/"history.sqlite3"
    r=run_experiment(ExperimentConfig(drift_levels=2,repeats=1,n_train=80,n_adapt=60,n_test=100))
    save_result(r,path)
    loaded=load_result(r.experiment_id,path)
    assert loaded["detail"].shape[0]==r.detail.shape[0]
    assert len(list_experiments(path))==1
    payload=json.loads(export_json(r))
    assert payload["experiment_id"]==r.experiment_id and "detail" in payload
    assert "Методика" in markdown_report(r)


def test_markdown_report_does_not_require_tabulate(monkeypatch):
    """Markdown export should work with only the declared pandas dependency."""
    from src import reporting
    monkeypatch.setattr(__import__("pandas").DataFrame, "to_markdown", lambda *a, **k: (_ for _ in ()).throw(AssertionError("to_markdown must not be used")))
    r = run_experiment(ExperimentConfig(drift_levels=2, repeats=1, n_train=80, n_adapt=60, n_test=100))
    report = reporting.markdown_report(r)
    assert "| intensity | strategy |" in report
    assert "| --- | --- |" in report
    assert "Средняя F1-мера" in report
