import tomllib
from pathlib import Path

import duckdb
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


def get_parquet_key():
    try:
        return st.secrets["encryption"]["parquet_key"]
    except Exception:
        import os

        return os.environ["PARQUET_KEY"]


parquet_key = get_parquet_key()


def load_encrypted_parquet(path, parquet_key):
    con = duckdb.connect()
    con.execute(f"PRAGMA add_parquet_key('footer_key', '{parquet_key}');")
    # Use read_parquet with encryption_config
    df = con.execute(
        f"SELECT * FROM read_parquet('{path}', encryption_config={{'footer_key': 'footer_key'}})"
    ).df()
    con.close()
    return df


st.set_page_config(page_title=app_conf["page_title"], layout="wide")
st.title(app_conf["title"])


# --- Sidebar: Mock Authentication ---
with st.sidebar:
    st.header("Authentication")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        if st.button("Login"):
            st.session_state["authenticated"] = True
            st.success("You are now logged in.")
        else:
            st.info("Please log in to use the app.")
    else:
        st.success("You are logged in.")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.info("You have been logged out.")

# --- Main Content: Only show if authenticated ---
if not st.session_state["authenticated"]:
    st.stop()  # Prevents the rest of the app from running if not authenticated


tab1, tab2, tab3 = st.tabs(
    [seg_conf["tab_name"], demand_conf["tab_name"], raw_conf["tab_name"]]
)


# --- Helper: Load scaler fit on training data (recommended for production) ---
def fit_scaler_on_training_data(data_path, features, parquet_key):
    df = load_encrypted_parquet(data_path, parquet_key)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df[features])
    return scaler


# --- Segmentation Tab ---
with tab1:
    st.header(seg_conf["tab_name"])
    if not SEG_MODEL_PATH.exists():
        st.error("Segmentation model not found. Please train and save the model first.")
    elif not CLIENT_DATA_PATH.exists():
        st.error(f"Client data file not found at: {CLIENT_DATA_PATH}")
    else:
        kmeans = mlflow.sklearn.load_model(str(SEG_MODEL_PATH))
        scaler = fit_scaler_on_training_data(
            CLIENT_DATA_PATH, SEG_FEATURES, parquet_key
        )

        st.subheader("Segment Uploaded Clients")
        uploaded = st.file_uploader("Upload client data (CSV)", type="csv", key="seg")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())
            if not all(col in df.columns for col in SEG_FEATURES):
                st.error(f"Data must contain columns: {SEG_FEATURES}")
            else:
                X_scaled = scaler.transform(df[SEG_FEATURES])
                clusters = kmeans.predict(X_scaled)
                df["cluster"] = clusters
                st.success("Segmentation complete.")
                st.write(df[["client_id", "cluster"]].head())
                st.download_button(
                    "Download Results as CSV",
                    df.to_csv(index=False),
                    file_name=seg_conf["download_filename"],
                )

        st.divider()
        st.subheader("Segment a Single Client (Manual Entry)")
        with st.form(key="segmentation_form"):
            age = st.number_input(
                "Age",
                min_value=seg_conf["age_min"],
                max_value=seg_conf["age_max"],
                value=30,
            )
            service_count = st.number_input(
                "Service Count",
                min_value=seg_conf["service_count_min"],
                max_value=seg_conf["service_count_max"],
                value=3,
            )
            seg_submit = st.form_submit_button("Predict Segment")
            if seg_submit:
                input_df = pd.DataFrame([[age, service_count]], columns=SEG_FEATURES)
                X_scaled = scaler.transform(input_df)
                cluster = kmeans.predict(X_scaled)[0]
                st.success(f"Predicted client segment: {cluster}")

# --- Demand Prediction Tab ---
with tab2:
    st.header(demand_conf["tab_name"])
    if not DEMAND_MODEL_PATH.exists():
        st.error(
            "Demand prediction model not found. Please train and save the model first."
        )
    elif not SERVICE_DATA_PATH.exists():
        st.error(f"Service data file not found at: {SERVICE_DATA_PATH}")
    else:
        logreg = mlflow.sklearn.load_model(str(DEMAND_MODEL_PATH))
        scaler = fit_scaler_on_training_data(
            SERVICE_DATA_PATH, DEMAND_FEATURES, parquet_key
        )

        st.subheader("Predict Demand for Uploaded Clients")
        uploaded2 = st.file_uploader(
            "Upload service data (CSV)", type="csv", key="demand"
        )
        if uploaded2:
            df2 = pd.read_csv(uploaded2)
            st.write("Preview:", df2.head())
            if not all(col in df2.columns for col in DEMAND_FEATURES):
                st.error(f"Data must contain columns: {DEMAND_FEATURES}")
            else:
                X_scaled = scaler.transform(df2[DEMAND_FEATURES])
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

        st.divider()
        st.subheader("Predict Demand for a Single Client (Manual Entry)")
        with st.form(key="demand_form"):
            age = st.number_input(
                "Age",
                min_value=demand_conf["age_min"],
                max_value=demand_conf["age_max"],
                value=30,
            )
            sessions_completed = st.number_input(
                "Sessions Completed",
                min_value=demand_conf["sessions_completed_min"],
                max_value=demand_conf["sessions_completed_max"],
                value=2,
            )
            demand_submit = st.form_submit_button("Predict Demand")
            if demand_submit:
                input_df = pd.DataFrame(
                    [[age, sessions_completed]], columns=DEMAND_FEATURES
                )
                X_scaled = scaler.transform(input_df)
                proba = logreg.predict_proba(X_scaled)[0, 1]
                pred = logreg.predict(X_scaled)[0]
                st.success(
                    f"Predicted high demand: {'Yes' if pred else 'No'} (probability: {proba:.2f})"
                )

# --- Raw Data Tab ---
with tab3:
    st.header(raw_conf["tab_name"])
    if not CLIENT_DATA_PATH.exists() or not SERVICE_DATA_PATH.exists():
        st.error(
            "One or more data files not found. Please ensure both client and service data exist."
        )
    else:
        df_clients = load_encrypted_parquet(CLIENT_DATA_PATH, parquet_key)
        df_services = load_encrypted_parquet(SERVICE_DATA_PATH, parquet_key)
        st.subheader("Client Data")
        st.dataframe(df_clients)
        st.markdown("---")
        st.subheader("Service Data")
        st.dataframe(df_services)
