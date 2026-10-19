import numpy as np
from src.models import fit_model, predict_model

def test_fit_predict():
    rng=np.random.default_rng(2); X=rng.normal(size=(150,2)); y=(X[:,0]+X[:,1]>0).astype(int)
    model=fit_model(X,y)
    pred=predict_model(model,X)
    assert pred.shape==(150,) and set(np.unique(pred)).issubset({0,1})

def test_pipeline_scaler_is_fitted_on_training_data():
    X=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
    y=np.array([0,0,1,1])
    model=fit_model(X,y)
    assert np.allclose(model.named_steps["scaler"].mean_,X.mean(axis=0))
