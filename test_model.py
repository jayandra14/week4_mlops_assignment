from sklearn import metrics
import pytest
import joblib
import pandas as pd

@pytest.fixture()
def data():
    data = pd.read_csv("./data/iris.csv")
    return data

@pytest.fixture()
def trained_model():
    model = joblib.load("./artifacts/models/v1/model.joblib") 
    return model

def test_data_no_null_values(data):
    assert data.isnull().sum().sum() == 0

def test_has_required_columns(data):
    required_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    assert required_columns.issubset(set(data.columns))

def test_model_loaded(trained_model):
    assert trained_model is not None

def test_model_prediction(trained_model, data):
    X = data[['sepal_length','sepal_width','petal_length','petal_width']]
    y = data.species
    y_pred = trained_model.predict(X)
    assert len(y_pred) == len(y)

def test_model_accuracy(trained_model, data):
    X = data[['sepal_length','sepal_width','petal_length','petal_width']]
    y = data.species
    accuracy = metrics.accuracy_score(y, trained_model.predict(X))
    assert accuracy > 0.90
