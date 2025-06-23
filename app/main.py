import tomllib
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import streamlit as st

with open("conf/config.toml", "rb") as f:
    config = tomllib.load(f)
ds = config["data_sources"]
seg_conf = config["segmentation"]
demand_conf = config["demand"]
raw_conf = config["raw_data"]
app_conf = config["app"]

SEG_MODEL_PATH = Path(ds["segmentation_model_path"])
DEMAND_MODEL_PATH = Path(ds["demand_model_path"])
CLIENT_DATA_PATH = Path(ds["client_data_path"])
SERVICE_DATA_PATH = Path(ds["service_data_path"])

SEG_FEATURES = seg_conf["features"]
DEMAND_FEATURES = demand_conf["features"]

st.set_page_config(page_title=app_conf["page_title"], layout="wide")
st.title(app_conf["title"])

tab1, tab2, tab3 = st.tabs(
    [seg_conf["tab_name"], demand_conf["tab_name"], raw_conf["tab_name"]]
)

with tab1:
    st.header(seg_conf["tab_name"])
    if not SEG_MODEL_PATH.exists():
        st.error("Segmentation model not found. Please train and save the model first.")
    elif not CLIENT_DATA_PATH.exists():
        st.error(f"Client data file not found at: {CLIENT_DATA_PATH}")
    else:
        df = pd.read_csv(CLIENT_DATA_PATH)
        st.write("Preview:", df.head())
        if not all(col in df.columns for col in SEG_FEATURES):
            st.error(f"Data must contain columns: {SEG_FEATURES}")
        else:
            try:
                kmeans = mlflow.sklearn.load_model(str(SEG_MODEL_PATH))
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df[SEG_FEATURES])
                clusters = kmeans.predict(X_scaled)
                df["cluster"] = clusters
                st.success("Segmentation complete.")
                st.write(df[["client_id", "cluster"]].head())
                st.download_button(
                    "Download Results as CSV",
                    df.to_csv(index=False),
                    file_name=seg_conf["download_filename"],
                )
            except Exception as e:
                st.error(f"Error loading model or predicting: {e}")

with tab2:
    st.header(demand_conf["tab_name"])
    if not DEMAND_MODEL_PATH.exists():
        st.error(
            "Demand prediction model not found. Please train and save the model first."
        )
    elif not SERVICE_DATA_PATH.exists():
        st.error(f"Service data file not found at: {SERVICE_DATA_PATH}")
    else:
        df2 = pd.read_csv(SERVICE_DATA_PATH)
        st.write("Preview:", df2.head())
        if not all(col in df2.columns for col in DEMAND_FEATURES):
            st.error(f"Data must contain columns: {DEMAND_FEATURES}")
        else:
            try:
                logreg = mlflow.sklearn.load_model(str(DEMAND_MODEL_PATH))
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df2[DEMAND_FEATURES])
                probs = logreg.predict_proba(X_scaled)[:, 1]
                preds = logreg.predict(X_scaled)
                df2["high_demand_proba"] = probs
                df2["high_demand_pred"] = preds
                st.success("Demand prediction complete.")
                st.write(
                    df2[["client_id", "high_demand_pred", "high_demand_proba"]].head()
                )
                st.download_button(
                    "Download Results as CSV",
                    df2.to_csv(index=False),
                    file_name=demand_conf["download_filename"],
                )
            except Exception as e:
                st.error(f"Error loading model or predicting: {e}")

with tab3:
    st.header(raw_conf["tab_name"])
    if not CLIENT_DATA_PATH.exists() or not SERVICE_DATA_PATH.exists():
        st.error(
            "One or more data files not found. Please ensure both client and service data exist."
        )
    else:
        df_clients = pd.read_csv(CLIENT_DATA_PATH)
        df_services = pd.read_csv(SERVICE_DATA_PATH)
        st.subheader("Client Data")
        st.dataframe(df_clients)
        st.subheader("Service Data")
        st.dataframe(df_services)
