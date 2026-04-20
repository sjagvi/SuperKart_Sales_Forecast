# Trains an XGBoost regressor on the SuperKart splits, tracks runs with MLflow,
# and pushes the best model to the Hugging Face model hub.

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ---- MLflow setup ----
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Sales-Forecast")

# ---- load splits ----
api = HfApi(token=os.getenv("HF_TOKEN"))
base = "hf://datasets/iamsubha/superkart"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest  = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"{base}/ytest.csv").squeeze()
print("Train:", Xtrain.shape, "Test:", Xtest.shape)

# ---- column groups ----
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Age',
]
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Product_Category',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
]

# ---- preprocessing + model pipeline ----
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
)

xgb_reg = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
pipe = make_pipeline(preprocessor, xgb_reg)

# small-ish grid - keeping CI runs reasonable
param_grid = {
    'xgbregressor__n_estimators': [100, 200],
    'xgbregressor__max_depth':    [3, 5],
    'xgbregressor__learning_rate':[0.05, 0.1],
    'xgbregressor__subsample':    [0.8, 1.0],
}

with mlflow.start_run():
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    mlflow.log_params(grid.best_params_)
    best = grid.best_estimator_

    # predict
    pred_tr = best.predict(Xtrain)
    pred_te = best.predict(Xtest)

    # metrics
    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(ytrain, pred_tr))),
        "train_mae":  float(mean_absolute_error(ytrain, pred_tr)),
        "train_r2":   float(r2_score(ytrain, pred_tr)),
        "test_rmse":  float(np.sqrt(mean_squared_error(ytest, pred_te))),
        "test_mae":   float(mean_absolute_error(ytest, pred_te)),
        "test_r2":    float(r2_score(ytest, pred_te)),
    }
    mlflow.log_metrics(metrics)
    print(metrics)

    # save model locally
    model_path = "best_superkart_model_v1.joblib"
    joblib.dump(best, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # push to HF model hub
    repo_id = "iamsubha/superkart-sales-model"
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Model repo '{repo_id}' exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print("Model uploaded.")
