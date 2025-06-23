import os
import tomllib
from pathlib import Path

import duckdb
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

with open("conf/config.toml", "rb") as f:
    CONFIG = tomllib.load(f)

ds = CONFIG["data_sources"]
seg_conf = CONFIG["segmentation"]
demand_conf = CONFIG["demand"]


def get_parquet_key():
    try:
        import streamlit as st

        return st.secrets["encryption"]["parquet_key"]
    except Exception:
        return os.environ["PARQUET_KEY"]


def load_encrypted_parquet(path, parquet_key):
    con = duckdb.connect()
    con.execute(f"PRAGMA add_parquet_key('footer_key', '{parquet_key}');")
    # Use read_parquet with encryption_config
    df = con.execute(
        f"SELECT * FROM read_parquet('{path}', encryption_config={{'footer_key': 'footer_key'}})"
    ).df()
    con.close()
    return df


def train_segmentation(df: pd.DataFrame, n_clusters: int = 4):
    features = seg_conf["features"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, clusters)
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_metric("silhouette_score", silhouette)
    # Prepare input_example and signature
    input_example = pd.DataFrame(X_scaled, columns=features).head(1)
    signature = infer_signature(X_scaled)
    # Log model with name, input_example, and signature
    mlflow.sklearn.log_model(
        model,
        name="segmentation_model",
        input_example=input_example,
        signature=signature,
    )
    Path(ds["segmentation_model_path"]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        model,
        ds["segmentation_model_path"],
        input_example=input_example,
        signature=signature,
    )
    print("Segmentation model trained and saved.")


def train_demand(df: pd.DataFrame):
    features = demand_conf["features"]
    target = "high_demand"
    df[target] = (df["sessions_needed"] > 5).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df[target]
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    roc_auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    mlflow.log_metric("roc_auc", roc_auc)
    # Prepare input_example and signature
    input_example = pd.DataFrame(X_scaled, columns=features).head(1)
    signature = infer_signature(X_scaled, model.predict(X_scaled))
    # Log model with name, input_example, and signature
    mlflow.sklearn.log_model(
        model,
        name="logistic_regression_model",
        input_example=input_example,
        signature=signature,
    )
    Path(ds["demand_model_path"]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        model, ds["demand_model_path"], input_example=input_example, signature=signature
    )
    print("Demand prediction model trained and saved.")


def main() -> None:
    mlflow.set_experiment("CommunityCare_Modeling")
    clients_df = load_encrypted_parquet(ds["client_data_path"], get_parquet_key())
    services_df = load_encrypted_parquet(ds["service_data_path"], get_parquet_key())
    with mlflow.start_run(run_name="segmentation_kmeans"):
        train_segmentation(clients_df, n_clusters=4)
    with mlflow.start_run(run_name="logistic_regression_demand"):
        train_demand(services_df)


if __name__ == "__main__":
    main()
