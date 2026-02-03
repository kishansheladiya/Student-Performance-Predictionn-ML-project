"""
Train models to predict student final grade (G3).
Saves the best model pipeline and metadata to /models.
"""
import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from download_data import download_and_extract

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "raw")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    # Use student-mat.csv
    csv_path = os.path.join(DATA_DIR, "student-mat.csv")
    if not os.path.exists(csv_path):
        download_and_extract()
    df = pd.read_csv(csv_path, sep=';')
    return df


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor


def evaluate_model(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def main():
    df = load_data()
    # target
    target = "G3"

    # quick EDA: ensure no missing target
    df = df.dropna(subset=[target])

    # choose features (a subset to keep the form short)
    candidate_features = [
        'sex', 'age', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
        'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
    ]
    df = df[[*candidate_features, target]].copy()

    X = df.drop(columns=[target])
    y = df[target]

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # feature lists
    numeric_features = ['age', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    categorical_features = [c for c in candidate_features if c not in numeric_features]

    preprocessor = build_pipeline(numeric_features, categorical_features)

    # models to evaluate
    models = {
        "linear": Pipeline([("preprocessor", preprocessor), ("reg", LinearRegression())]),
        "rf": Pipeline([("preprocessor", preprocessor), ("reg", RandomForestRegressor(random_state=42))]),
        "gbr": Pipeline([("preprocessor", preprocessor), ("reg", GradientBoostingRegressor(random_state=42))]),
    }

    param_distributions = {
        "rf": {
            'reg__n_estimators': [100, 200],
            'reg__max_depth': [None, 5, 10],
        },
        "gbr": {
            'reg__n_estimators': [100, 200],
            'reg__learning_rate': [0.01, 0.1],
            'reg__max_depth': [3, 5],
        }
    }

    results = {}

    for name, pipe in models.items():
        print(f"Training {name}...")
        if name in param_distributions:
            search = RandomizedSearchCV(pipe, param_distributions[name], n_iter=4, cv=3, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
        else:
            pipe.fit(X_train, y_train)
            best = pipe

        y_pred = best.predict(X_val)
        metrics = evaluate_model(y_val, y_pred)
        results[name] = {"metrics": metrics}
        print(f"{name} metrics:", metrics)

    # pick best by rmse
    best_name = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
    print("Best model:", best_name)

    # retrain best model on full data
    best_pipeline = models[best_name]
    if best_name in param_distributions:
        # reuse RandomizedSearch on full data for better fit
        search = RandomizedSearchCV(models[best_name], param_distributions[best_name], n_iter=8, cv=3, random_state=42, n_jobs=-1)
        search.fit(X, y)
        best_pipeline = search.best_estimator_
    else:
        best_pipeline.fit(X, y)

    # metadata
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_name = f"student_performance_{best_name}"
    model_version = f"v1-{timestamp}"
    model_filename = f"{model_name}_{model_version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(best_pipeline, model_path)

    final_metrics = evaluate_model(y, best_pipeline.predict(X))

    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "best_algo": best_name,
        "metrics": final_metrics,
        "features": list(X.columns),
    }

    meta_path = os.path.join(MODEL_DIR, f"{model_name}_{model_version}.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Saved model to", model_path)
    print("Saved metadata to", meta_path)


if __name__ == "__main__":
    main()
