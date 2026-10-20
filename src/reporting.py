import json
import pandas as pd

def export_csv(detail, aggregate):
    return detail.to_csv(index=False), aggregate.to_csv(index=False)

def export_paired_csv(detail):
    from .statistics import paired_comparisons
    return paired_comparisons(detail).to_csv(index=False)

def export_json(result):
    return json.dumps({
        "format_version": 1, "experiment_id": result.experiment_id,
        "created_at": result.created_at, "config": result.config,
        "detail": json.loads(result.detail.to_json(orient="records")),
        "aggregate": json.loads(result.aggregate.to_json(orient="records"))
    }, ensure_ascii=False, indent=2, allow_nan=False, default=str)

def _markdown_cell(value):
    """Render a value safely inside a Markdown table cell without optional deps."""
    if pd.isna(value):
        return "—"
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return text


def _markdown_table(frame):
    """Create a Markdown table using pandas only (no tabulate dependency)."""
    if frame.empty:
        return "Нет данных."
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(_markdown_cell(c) for c in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def markdown_report(result):
    cfg=result.config
    top = result.aggregate
    metric="f1_mean"
    lines=["# DriftLab — отчёт эксперимента", "", f"- ID: `{result.experiment_id}`",
           f"- Дата UTC: {result.created_at}", f"- Тип дрейфа: {cfg['drift_type']}",
           f"- Модель: {cfg['model']}", f"- Повторы: {cfg['repeats']}",
           f"- Выборки: train={cfg['n_train']}, adapt={cfg['n_adapt']}, test={cfg['n_test']}",
           f"- Шум меток: {cfg['label_noise']}, seed: {cfg['seed']}", "",
           "## Методика", "",
           "Сравнивались исходная модель, переобучение на новых данных и обучение на объединении старых и новых данных. Все стратегии оценивались на общей тестовой выборке внутри каждого повтора и уровня дрейфа.",
           "Тестовые выборки независимы от обучающих. Данные синтетические; результаты не являются оценкой производственной системы.", "",
           "## Средняя F1-мера", ""]
    if not top.empty:
        cols=[c for c in ["intensity","strategy","f1_mean","f1_std","n_successful_repeats"] if c in top]
        lines.append(_markdown_table(top[cols].round(4)))
    else: lines.append("Нет агрегированных результатов.")
    from .statistics import paired_comparisons
    paired = paired_comparisons(result.detail)
    lines += ["", "## Парные сравнения стратегий", "",
              "Разность рассчитана как первая стратегия минус вторая; 95%-й интервал построен по распределению Стьюдента. Поправка Холма применяется отдельно для каждой метрики по всем парам стратегий и уровням интенсивности.", ""]
    if not paired.empty:
        cols = [c for c in ["metric", "intensity", "contrast", "mean_diff", "ci95_low", "ci95_high", "p_raw", "p_holm", "reject_holm_0_05"] if c in paired]
        lines.append(_markdown_table(paired[cols].round(5)))
    lines += ["", "## Ограничения", "",
              "- 95%-е интервалы и парные t-тесты описывают вариативность между повторами; выводы относятся к выбранной синтетической постановке.",
              "- Covariate shift задаётся смещением второго признака с последующей разметкой фиксированным нелинейным правилом: меняется P(X), правило P(Y|X) остаётся прежним.",
              "- Concept drift меняет ориентацию правила разметки; модели и результаты относятся к заданному синтетическому сценарию.", ""]
    return "\n".join(lines)
