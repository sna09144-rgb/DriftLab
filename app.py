import streamlit as st
import pandas as pd
from src.config import ExperimentConfig, STRATEGIES, METRIC_LABELS
from src.experiment import run_experiment
from src.storage import save_result, list_experiments, load_result
from src.statistics import paired_difference
from src.reporting import export_csv, export_json, markdown_report
from src.visualization import drift_chart, metric_comparison, scatter_plot, decision_boundary, confusion_heatmap
from src.metrics import calculate_metrics
from src.models import predict_model, positive_scores

st.set_page_config(page_title="DriftLab", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background:#101722;color:#e7edf5}
[data-testid="stSidebar"] {background:#172231}
.block-container {padding-top:1.7rem;max-width:1500px}
h1,h2,h3 {color:#edf3fb}
div[data-testid="stMetric"] {background:#1a2635;border:1px solid #2c3b4e;padding:14px;border-radius:12px}
.stButton>button {border-radius:9px;border:1px solid #405774}
small, .stCaption {color:#9aaec3}
</style>""", unsafe_allow_html=True)

st.title("◈ DriftLab")
st.caption("Interactive Machine Learning Drift Research Laboratory · Синтетическая исследовательская среда")
pages=["Главная","Лаборатория","Сравнение и аналитика","История","Методология","О проекте"]
page=st.sidebar.radio("Раздел",pages)
st.sidebar.markdown("---")
st.sidebar.caption("Результаты рассчитываются при запуске эксперимента. Демонстрационные данные не подставляются автоматически.")

def latest_result():
    return st.session_state.get("result")

if page=="Главная":
    st.subheader("Исследование влияния дрейфа данных на качество моделей")
    st.write("Изучайте, как изменение распределения признаков и изменение правила классификации влияют на модель, и сравнивайте стратегии адаптации.")
    result=latest_result()
    history=list_experiments()
    a,b,c=st.columns(3)
    a.metric("Сохранённых экспериментов",len(history))
    b.metric("Последний запуск","В текущей сессии" if result else ("Доступен в истории" if history else "Нет запусков"))
    c.metric("Научные сценарии","2")
    if result:
        st.success(f"Последний запуск: {result.experiment_id}")
        st.plotly_chart(drift_chart(result.aggregate,"f1"),use_container_width=True)
    elif history:
        st.info("Есть сохранённые запуски. Откройте раздел «История», чтобы загрузить результат.")
    else:
        st.info("Экспериментов пока нет. Откройте «Лаборатория» и запустите первый расчёт.")
    st.markdown("**Быстрый старт:** выберите тип дрейфа → настройте объём выборок и повторы → запустите эксперимент → изучите метрики → экспортируйте результаты.")

elif page=="Лаборатория":
    st.subheader("Настройка эксперимента")
    demo=st.toggle("Демо-режим (облегчённый настоящий эксперимент)",value=False)
    with st.form("experiment_form"):
        drift_label=st.radio("Сценарий дрейфа",["Covariate shift — изменение P(X)","Concept drift — изменение P(Y|X)"])
        col1,col2,col3=st.columns(3)
        with col1:
            dmin=st.number_input("Минимальная интенсивность",0.0,1.0,0.0,.05)
            dmax=st.number_input("Максимальная интенсивность",0.0,1.0,1.0,.05)
            levels=st.slider("Уровней дрейфа",2,11,4)
        with col2:
            repeats=st.select_slider("Повторы",options=[1,2,3,5,10,20,30],value=3 if demo else 5)
            ntrain=st.number_input("Исходная обучающая выборка",100,10000,300 if demo else 500,100)
            nadapt=st.number_input("Новые данные для адаптации",100,10000,200 if demo else 300,100)
        with col3:
            ntest=st.number_input("Тестовая выборка",100,20000,500 if demo else 1000,100)
            noise=st.slider("Шум меток",0.0,0.4,0.05,.01)
            seed=st.number_input("Случайный seed",0,2147483647,42)
        chosen=st.multiselect("Стратегии",list(STRATEGIES),default=list(STRATEGIES),format_func=lambda x:STRATEGIES[x])
        submitted=st.form_submit_button("▶ Запустить эксперимент",type="primary",use_container_width=True)
    if submitted:
        try:
            cfg=ExperimentConfig(drift_type="covariate" if drift_label.startswith("Covariate") else "concept",
                drift_min=float(dmin),drift_max=float(dmax),drift_levels=int(levels),
                repeats=int(repeats),n_train=int(ntrain),n_adapt=int(nadapt),n_test=int(ntest),
                label_noise=float(noise),seed=int(seed),strategies=tuple(chosen)).validate()
            if demo:
                cfg=ExperimentConfig(drift_type=cfg.drift_type,drift_min=cfg.drift_min,drift_max=cfg.drift_max,
                    drift_levels=min(cfg.drift_levels,4),repeats=min(cfg.repeats,3),n_train=min(cfg.n_train,300),
                    n_adapt=min(cfg.n_adapt,200),n_test=min(cfg.n_test,500),label_noise=cfg.label_noise,
                    seed=cfg.seed,strategies=cfg.strategies).validate()
                st.info("Демо-режим: синтетические данные, ограниченное число уровней и повторов.")
            progress=st.progress(0.0,text="Подготовка эксперимента")
            result=run_experiment(cfg,progress_callback=lambda p:progress.progress(p,text=f"Выполнено: {p:.0%}"))
            save_result(result)
            st.session_state["result"]=result
            st.success(f"Эксперимент завершён и сохранён. ID: {result.experiment_id}")
        except Exception as e:
            st.error(f"Эксперимент не выполнен: {e}")
    result=latest_result()
    if result:
        st.markdown("### Результаты текущего запуска")
        metric=st.selectbox("Метрика главного графика",list(METRIC_LABELS),index=3,format_func=lambda x:METRIC_LABELS[x],key="main_metric")
        st.plotly_chart(drift_chart(result.aggregate,metric),use_container_width=True)
        st.dataframe(result.aggregate,use_container_width=True)
        d,a=export_csv(result.detail,result.aggregate)
        st.download_button("Скачать детальные результаты CSV",d,"driftlab_detail.csv","text/csv")
        st.download_button("Скачать агрегированные результаты CSV",a,"driftlab_aggregate.csv","text/csv")
        st.download_button("Экспорт JSON",export_json(result),"driftlab_experiment.json","application/json")
        st.download_button("Научный отчёт Markdown",markdown_report(result),"driftlab_report.md","text/markdown")

elif page=="Сравнение и аналитика":
    st.subheader("Сравнение стратегий и диагностика")
    result=latest_result()
    if not result:
        st.info("Сначала запустите эксперимент или загрузите его из раздела «История».")
    else:
        metric=st.selectbox("Метрика",list(METRIC_LABELS),index=3,format_func=lambda x:METRIC_LABELS[x])
        st.plotly_chart(drift_chart(result.aggregate,metric),use_container_width=True)
        intensities=sorted(result.aggregate["intensity"].unique())
        selected=st.select_slider("Уровень дрейфа для сравнения метрик",options=intensities,value=intensities[len(intensities)//2])
        st.plotly_chart(metric_comparison(result.aggregate,selected),use_container_width=True)
        st.markdown("### Парная разность стратегий (по повторам, t-интервал 95%)")
        strats=[s for s in STRATEGIES if s in set(result.detail.strategy)]
        if len(strats)>=2:
            ca,cb=st.columns(2)
            sa=ca.selectbox("Стратегия A",strats,index=len(strats)-1,format_func=lambda x:STRATEGIES[x])
            sb=cb.selectbox("Стратегия B",strats,index=0,format_func=lambda x:STRATEGIES[x])
            if sa!=sb:
                st.dataframe(paired_difference(result.detail,sa,sb,metric),use_container_width=True)
                st.caption("Разность A − B на одной тестовой выборке. «excludes_zero» — описательный признак, не поправка на множественные сравнения.")
        st.dataframe(result.detail,use_container_width=True)
        ex=result.examples
        if ex and "X" in ex:
            left,right=st.columns(2)
            with left: st.plotly_chart(scatter_plot(ex["reference_X"],ex["reference_y"],"Исходное распределение"),use_container_width=True)
            with right: st.plotly_chart(scatter_plot(ex["X"],ex["y"],f"Тестовые данные, интенсивность {ex['intensity']:.2f}"),use_container_width=True)
            st.plotly_chart(decision_boundary(ex["baseline_model"],ex["X"],ex["y"],"Граница решений исходной модели (средний уровень дрейфа, повтор 1)"),use_container_width=True)
            pred=predict_model(ex["baseline_model"],ex["X"]); score=positive_scores(ex["baseline_model"],ex["X"])
            st.plotly_chart(confusion_heatmap(calculate_metrics(ex["y"],pred,score)),use_container_width=True)
        # Data-derived interpretation, without claims of statistical significance.
        means=result.aggregate
        key=f"{metric}_mean"
        if key in means and not means.empty:
            st.markdown("### Автоматическая интерпретация")
            for intensity, group in means.groupby("intensity"):
                group=group.dropna(subset=[key]).sort_values(key,ascending=False)
                if len(group):
                    winner=group.iloc[0]
                    baseline=group[group.strategy=="baseline"]
                    delta=float(winner[key]-baseline.iloc[0][key]) if len(baseline) else float("nan")
                    st.write(f"При интенсивности **{intensity:.2f}** наибольшее среднее значение {METRIC_LABELS[metric]} у стратегии **{STRATEGIES.get(winner.strategy,winner.strategy)}** ({winner[key]:.3f}). Разница с исходной моделью: {delta:+.3f}. Это описательное сравнение, не тест статистической значимости.")

elif page=="История":
    st.subheader("Сохранённые эксперименты")
    history=list_experiments()
    if not history: st.info("История пуста. Результаты появятся после успешного запуска.")
    else:
        hist=pd.DataFrame([{"ID":h["experiment_id"],"Создан":h["created_at"],"Дрейф":h["config"].get("drift_type"),
            "Повторы":h["config"].get("repeats"),"Seed":h["config"].get("seed"),"Формат":h["format_version"]} for h in history])
        st.dataframe(hist,use_container_width=True)
        selected=st.selectbox("Выберите эксперимент",options=[h["experiment_id"] for h in history])
        if st.button("Загрузить выбранный эксперимент"):
            try:
                saved=load_result(selected)
                # Historical detail and aggregate can be inspected/exported; examples/models are intentionally not serialized.
                from src.experiment import ExperimentResult
                result=ExperimentResult(saved["experiment_id"],saved["created_at"],saved["config"],saved["detail"],saved["aggregate"],{})
                st.session_state["result"]=result
                st.success("Результат загружен в текущую сессию. Для повторного просмотра откройте «Сравнение и аналитика».")
            except Exception as e: st.error(f"Не удалось загрузить результат: {e}")
        st.caption("История хранится локально в results/driftlab.sqlite3. Удаление автоматически не выполняется.")

elif page=="Методология":
    st.subheader("Методология")
    st.markdown("""
### Covariate shift
Генератор сначала создаёт двумерные признаки, затем применяет контролируемое преобразование: смещение второго признака на \\(1.5d\\). Метки после преобразования генерируются одним и тем же нелинейным условным правилом \\(P(Y\\mid X)=\\sigma(2.4\\,X\\cdot w + 1.5(x_1^2-x_2^2))\\) при всех уровнях дрейфа. Нелинейность важна: логистическая регрессия не описывает правило точно, поэтому сдвиг \\(P(X)\\) способен ухудшить качество. Поэтому меняется \\(P(X)\\), а механизм разметки остаётся фиксированным. Это синтетическая контролируемая постановка, а не доказательство равенства условных распределений на произвольных данных.

### Concept drift
Распределение признаков генерируется без преобразования covariate shift, а угол вектора правила меток меняется от \\(\\pi/4\\) до \\(3\\pi/4\\) в зависимости от интенсивности. Это меняет условный механизм \\(P(Y\\mid X)\\).

### Разделение выборок
Исходная обучающая выборка, новые данные адаптации и тестовая выборка генерируются с независимыми потоками seed. Тест не используется для обучения. В рамках каждого уровня и повтора все стратегии оцениваются на одной тестовой выборке.

### Стратегии
- **Исходная модель:** обучена только на исходных данных.
- **Переобучение на новых данных:** обучена только на новых данных адаптации.
- **Объединение:** обучена на исходных и новых данных. Объём обучающей выборки у неё больше; это явно отражено в результатах.

### Метрики и статистика
Положительный класс — 1. Precision/Recall/F1 используют `zero_division=0`. ROC-AUC недоступна, если тест содержит один класс. Средние и выборочные стандартные отклонения считаются по независимым повторам. 95%-й интервал строится по распределению Стьюдента (df = n−1). На графике пунктиром показан oracle — точность оптимального классификатора, знающего истинное правило (верхняя граница).

### Ограничения
Это двумерный синтетический эксперимент с логистической регрессией. Он не моделирует все виды дрейфа, не доказывает универсальное превосходство стратегий и не заменяет проверку на реальных данных.
""")

else:
    st.subheader("О проекте")
    st.write("DriftLab — локальная интерактивная лаборатория для воспроизводимого исследования дрейфа данных и адаптации моделей.")
    st.markdown("""
**Стек:** Python, NumPy, pandas, scikit-learn, Plotly, Streamlit, SQLite, pytest.

**Границы применимости:** синтетическая двумерная бинарная классификация; одна базовая модель — логистическая регрессия. Метрики описывают только выбранную генеративную постановку. Приложение не является универсальной системой обнаружения дрейфа.
""")
