import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .config import STRATEGIES, METRIC_LABELS

PALETTE={"baseline":"#7c9cff","recent":"#42d6b3","combined":"#ffbd69"}

def drift_chart(aggregate, metric="f1"):
    if aggregate.empty: return go.Figure().update_layout(title="Нет результатов")
    mean=f"{metric}_mean"; low=f"{metric}_ci95_low"; high=f"{metric}_ci95_high"
    fig=go.Figure()
    for strategy, group in aggregate.groupby("strategy"):
        group=group.sort_values("intensity")
        name=STRATEGIES.get(strategy,strategy)
        fig.add_trace(go.Scatter(x=group.intensity,y=group[mean],mode="lines+markers",name=name,
            line={"color":PALETTE.get(strategy)}, error_y=dict(type="data",
            symmetric=False, array=(group[high]-group[mean]).clip(lower=0).fillna(0),
            arrayminus=(group[mean]-group[low]).clip(lower=0).fillna(0))))
    if metric == "accuracy" and "oracle_accuracy_mean" in aggregate:
        o = aggregate.groupby("intensity")["oracle_accuracy_mean"].mean().reset_index().sort_values("intensity")
        fig.add_trace(go.Scatter(x=o.intensity, y=o.oracle_accuracy_mean, mode="lines", name="Oracle (предел)",
                                 line={"color": "#ffffff", "dash": "dash", "width": 1.5}))
    fig.update_layout(template="plotly_dark", xaxis_title="Интенсивность дрейфа",
        yaxis_title=METRIC_LABELS.get(metric,metric), legend_title="Стратегия", margin=dict(l=20,r=20,t=30,b=20))
    fig.update_yaxes(range=[0,1])
    return fig

def metric_comparison(aggregate, intensity=None):
    if aggregate.empty: return go.Figure()
    df=aggregate.copy()
    if intensity is not None: df=df[np.isclose(df.intensity,float(intensity))]
    metrics=[m for m in ["accuracy","precision","recall","f1","specificity","balanced_accuracy"] if f"{m}_mean" in df]
    long=[]
    for _,r in df.iterrows():
        for m in metrics: long.append({"Стратегия":STRATEGIES.get(r.strategy,r.strategy),"Метрика":METRIC_LABELS[m],"Значение":r[f"{m}_mean"]})
    fig=px.bar(pd.DataFrame(long),x="Метрика",y="Значение",color="Стратегия",barmode="group",template="plotly_dark")
    fig.update_yaxes(range=[0,1])
    return fig

def scatter_plot(X,y,title="Распределение тестовых данных"):
    fig=px.scatter(x=np.asarray(X)[:,0],y=np.asarray(X)[:,1],color=np.asarray(y).astype(str),
        labels={"x":"Признак 1","y":"Признак 2","color":"Класс"},title=title,template="plotly_dark",opacity=.75)
    return fig

def decision_boundary(model,X,y,title="Граница решений"):
    X=np.asarray(X); y=np.asarray(y)
    x0=np.linspace(X[:,0].min()-.5,X[:,0].max()+.5,150)
    x1=np.linspace(X[:,1].min()-.5,X[:,1].max()+.5,150)
    xx,yy=np.meshgrid(x0,x1)
    grid=np.c_[xx.ravel(),yy.ravel()]
    probs=model.predict_proba(grid)[:,1].reshape(xx.shape)
    fig=go.Figure()
    fig.add_trace(go.Contour(x=x0,y=x1,z=probs,showscale=False,contours=dict(start=.5,end=.5,size=.5),
        line=dict(color="#ffbd69",width=3),name="Граница P(Y=1)=0.5"))
    for cls,color in [(0,"#7c9cff"),(1,"#42d6b3")]:
        mask=y==cls
        fig.add_trace(go.Scatter(x=X[mask,0],y=X[mask,1],mode="markers",name=f"Класс {cls}",
            marker=dict(color=color,size=5,opacity=.65)))
    fig.update_layout(title=title,template="plotly_dark",xaxis_title="Признак 1",yaxis_title="Признак 2")
    return fig

def confusion_heatmap(metrics):
    z=np.array(metrics["confusion_matrix"])
    fig=px.imshow(z,text_auto=True,x=["Предсказан 0","Предсказан 1"],y=["Факт 0","Факт 1"],
        labels={"x":"Предсказанный класс","y":"Истинный класс","color":"Количество"},
        title="Матрица ошибок",template="plotly_dark",color_continuous_scale="Blues")
    return fig
