import pandas as pd
import os
import joblib
import argparse

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from datetime import datetime


def train_model_and_save(dataset_path, model_tag):
    data = pd.read_csv(dataset_path)
    
    train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
    X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train.species
    X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test.species
    
    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(X_train, y_train)
    
    train_acc = metrics.accuracy_score(y_train, mod_dt.predict(X_train))
    test_acc = metrics.accuracy_score(y_test, mod_dt.predict(X_test))

    model_path = f"artifacts/models/{model_tag}"
    os.makedirs(model_path, exist_ok=True)
    
    joblib.dump(mod_dt, f"{model_path}/model.joblib")
    with open(f"{model_path}/metrics.txt", "w") as f:
        f.write(f"train_acc: {train_acc}\n")
        f.write(f"test_acc: {test_acc}\n")
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset CSV file')
    argparser.add_argument('--model_tag', type=str, required=False, help='Tag for the model version')
    args = argparser.parse_args()
    
    train_model_and_save(args.dataset_path, args.model_tag)