import tomllib
from pathlib import Path

import duckdb
import mlflow.sklearn
import pandas as pd
import plotly.express as px
import streamlit as st


def show_cluster_spider_plot(df, features, clusters, cluster_col="segment"):
    """
    Display a radar (spider) plot showing normalized feature means for each cluster.
    """
    df = df.copy()
    df[cluster_col] = clusters
    means = df.groupby(cluster_col)[features].mean().reset_index()
    # Min-max scale each feature for visualization
    means_scaled = means.copy()
    for feature in features:
        min_val = means[feature].min()
        max_val = means[feature].max()
        if max_val > min_val:
            means_scaled[feature] = (means[feature] - min_val) / (max_val - min_val)
        else:
            means_scaled[feature] = 0.5  # If constant, set to mid-point

    # print(f"Cluster Feature Means (scaled):\n {means_scaled}")
    # print(features)

    melted = means_scaled.melt(
        id_vars=cluster_col, var_name="Feature", value_name="Scaled Mean"
    )
    fig = px.line_polar(
        melted,
        r="Scaled Mean",
        theta="Feature",
        color=cluster_col,
        line_close=True,
        markers=True,
        title="Cluster Profiles (Spider Plot, Scaled)",
    )
    fig.update_traces(fill="toself")
    # st.subheader("Cluster Profiles (Spider Plot, Scaled)")
    st.plotly_chart(fig, use_container_width=True)


def show_cluster_size_bubbles(
    cluster_sizes: pd.DataFrame,
    title: str = "Cluster Sizes (relative, bubble size = count, label = % of total)",
) -> px.scatter:
    total = cluster_sizes["count"].sum()
    cluster_sizes = cluster_sizes.copy()
    cluster_sizes["percent"] = 100 * cluster_sizes["count"] / total
    cluster_sizes["label"] = cluster_sizes.apply(
        lambda row: f"{row['segment']:.0f}: {row['percent']:.1f}% ({row['count']:.0f})",
        axis=1,
    )
    cluster_sizes["x"] = range(len(cluster_sizes))
    cluster_sizes["y"] = 0  # All on same horizontal line
    fig = px.scatter(
        cluster_sizes,
        x="x",
        y="y",
        size="count",
        color="segment",
        hover_name="segment",
        hover_data={"count": True, "percent": ":.0f"},
        text="label",
        size_max=80,
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=2, color="DarkSlateGrey")),
    )
    fig.update_layout(
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )
    return fig


def show_cluster_profiles(df, features, clusters, cluster_col="segment"):
    """Display cluster profiles as a summary table and bar chart."""
    df = df.copy()
    df[cluster_col] = clusters

    # Compute means for numeric features
    means = df.groupby(cluster_col)[features].mean().reset_index()
    st.subheader("Cluster Feature Means")
    st.dataframe(means)
    print(f"Cluster Feature Means:\n {means}")

    # Melt for plotting
    melted = means.melt(id_vars=cluster_col, var_name="Feature", value_name="Mean")
    fig = px.bar(
        melted,
        x="Feature",
        y="Mean",
        color=cluster_col,
        barmode="group",
        title="Feature Means by Cluster",
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Configuration Loader ---
class AppConfig:
    def __init__(self, config_path="conf/config.toml"):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        self.ds = self.config["data_sources"]
        self.seg_conf = self.config["segmentation"]
        self.demand_conf = self.config["demand"]
        self.app_conf = self.config["app"]


# --- Data Loader ---
class EncryptedParquetLoader:
    def __init__(self, parquet_key):
        self.parquet_key = parquet_key

    def load(self, path):
        con = duckdb.connect()
        con.execute(f"PRAGMA add_parquet_key('footer_key', '{self.parquet_key}');")
        df = con.execute(
            f"SELECT * FROM read_parquet('{path}', encryption_config={{'footer_key': 'footer_key'}})"
        ).df()
        con.close()
        return df


# --- Scaler Utility ---
class ScalerUtility:
    @staticmethod
    def fit_scaler(df, features):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(df[features])
        return scaler


# --- Authentication ---
class MockAuthenticator:
    def __init__(self):
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False

    def sidebar(self):
        with st.sidebar:
            st.header("Authentication")
            if not st.session_state["authenticated"]:
                if st.button("Login"):
                    st.session_state["authenticated"] = True
                    st.success("You are now logged in .")
                else:
                    st.info("Please log in to use the app.")
            else:
                st.success("You are logged in .")
                if st.button("Logout"):
                    st.session_state["authenticated"] = False
                    st.info("You have been logged out.")

    def require_auth(self):
        if not st.session_state["authenticated"]:
            st.stop()


# --- Main App ---
class CommunityCareApp:
    def __init__(self, config_path="conf/config.toml"):
        self.config = AppConfig(config_path)
        self.parquet_key = self.get_parquet_key()
        self.loader = EncryptedParquetLoader(self.parquet_key)
        self.auth = MockAuthenticator()

    @staticmethod
    def get_parquet_key():
        try:
            return st.secrets["encryption"]["parquet_key"]
        except Exception:
            import os

            return os.environ["PARQUET_KEY"]

    def run(self):
        st.set_page_config(page_title=self.config.app_conf["page_title"], layout="wide")
        self.auth.sidebar()
        self.auth.require_auth()
        st.title(self.config.app_conf["title"])

        # Top-level tabs for all sections
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                self.config.seg_conf["tab_name"],
                self.config.demand_conf["tab_name"],
                "Client Data",
                "Service Data",
            ]
        )
        self.segmentation_tab(tab1)
        self.demand_tab(tab2)
        self.show_data_tab(
            tab3,
            self.config.ds["client_data_path"],
            self.config.ds["clients_meta"],
            "Client",
        )
        self.show_data_tab(
            tab4,
            self.config.ds["service_data_path"],
            self.config.ds["services_meta"],
            "Service",
        )

    def segmentation_tab(self, tab):
        with tab:
            st.header(self.config.seg_conf["tab_name"])

            available_ks = [2, 3, 4, 5, 6]
            selected_k = st.selectbox(
                "Select number of clusters (k):", available_ks, index=2
            )

            model_path = Path(self.config.ds["segmentation_model_path"]).with_name(
                f"segmentation_model_k{selected_k}.pkl"
            )
            client_data_path = Path(self.config.ds["client_data_path"])
            categorical_features = self.config.seg_conf["categorical_features"]
            numeric_features = self.config.seg_conf["numeric_features"]

            if not model_path.exists():
                st.error(
                    f"Segmentation model for k={selected_k} not found at: {model_path}"
                )
                return

            if not client_data_path.exists():
                st.error(f"Client data file not found at: {client_data_path}")
                return

            # Load model, data, and scaler
            kmeans = mlflow.sklearn.load_model(str(model_path))
            df_clients = self.loader.load(client_data_path)
            scaler = ScalerUtility.fit_scaler(df_clients, numeric_features)

            # --- Prepare features for clustering ---
            # One-hot encode categoricals
            X = pd.get_dummies(
                df_clients[numeric_features + categorical_features], drop_first=True
            )
            # Save the columns used for one-hot encoding (should match training)
            fit_columns = X.columns.tolist()

            # Scale numeric features
            X[numeric_features] = scaler.transform(X[numeric_features])

            # --- Predict Segment for a Single Client (Manual Entry) ---
        st.subheader("Predict Segment for a Client")
        with st.form(key="segmentation_form"):
            cols = st.columns(2)
            input_dict = {}

            # Numeric fields in first column, categorical in second (or alternate as needed)
            for i, num_feat in enumerate(numeric_features):
                with cols[i % 2]:
                    input_dict[num_feat] = st.number_input(
                        num_feat.capitalize(),
                        min_value=float(df_clients[num_feat].min()),
                        max_value=float(df_clients[num_feat].max()),
                        value=float(df_clients[num_feat].mean()),
                    )
            for i, cat_feat in enumerate(categorical_features):
                with cols[i % 2]:
                    input_dict[cat_feat] = st.selectbox(
                        cat_feat.capitalize(),
                        df_clients[cat_feat].unique(),
                    )
            seg_submit = st.form_submit_button("Predict Segment")

            if seg_submit:
                input_df = pd.DataFrame([input_dict])
                # One-hot encode and align columns
                X_input = pd.get_dummies(input_df, drop_first=True)
                for col in fit_columns:
                    if col not in X_input.columns:
                        X_input[col] = 0
                X_input = X_input[fit_columns]
                # Scale numeric features
                X_input[numeric_features] = scaler.transform(X_input[numeric_features])
                cluster = kmeans.predict(X_input)[0]
                st.success(f"Predicted client segment: {cluster}")

            st.divider()

            # --- Display Cluster Sizes as Plotly Bubble Chart ---
            st.subheader("Segment Distribution")
            clusters = kmeans.predict(X)
            cluster_sizes = (
                pd.Series(clusters)
                .value_counts()
                .sort_index()
                .reset_index()
                .rename(columns={"index": "segment", 0: "count"})
            )
            fig = show_cluster_size_bubbles(cluster_sizes)
            st.plotly_chart(fig, use_container_width=True)

            show_cluster_profiles(X, X.columns, clusters)
            show_cluster_spider_plot(X, X.columns, clusters)

    def demand_tab(self, tab):
        with tab:
            st.header(self.config.demand_conf["tab_name"])
            demand_model_path = Path(self.config.ds["demand_model_path"])
            service_data_path = Path(self.config.ds["service_data_path"])
            demand_features = self.config.demand_conf["features"]

            if not demand_model_path.exists():
                st.error(
                    "Demand prediction model not found. Please train and save the model first."
                )
            elif not service_data_path.exists():
                st.error(f"Service data file not found at: {service_data_path}")
            else:
                logreg = mlflow.sklearn.load_model(str(demand_model_path))
                df_services = self.loader.load(service_data_path)
                scaler = ScalerUtility.fit_scaler(df_services, demand_features)

                st.subheader("Predict Demand for Uploaded Clients")
                uploaded2 = st.file_uploader(
                    "Upload service data (CSV)", type="csv", key="demand"
                )
                if uploaded2:
                    df2 = pd.read_csv(uploaded2)
                    st.write("Preview:", df2.head())
                    if not all(col in df2.columns for col in demand_features):
                        st.error(f"Data must contain columns: {demand_features}")
                    else:
                        X_scaled = scaler.transform(df2[demand_features])
                        probs = logreg.predict_proba(X_scaled)[:, 1]
                        preds = logreg.predict(X_scaled)
                        df2["high_demand_proba"] = probs
                        df2["high_demand_pred"] = preds
                        st.success("Demand prediction complete.")
                        st.write(
                            df2[
                                ["client_id", "high_demand_pred", "high_demand_proba"]
                            ].head()
                        )
                        st.download_button(
                            "Download Results as CSV",
                            df2.to_csv(index=False),
                            file_name=self.config.demand_conf["download_filename"],
                        )

                st.divider()
                st.subheader("Predict Demand for a Single Client (Manual Entry)")
                with st.form(key="demand_form"):
                    age = st.number_input(
                        "Age",
                        min_value=self.config.demand_conf["age_min"],
                        max_value=self.config.demand_conf["age_max"],
                        value=30,
                    )
                    sessions_completed = st.number_input(
                        "Sessions Completed",
                        min_value=self.config.demand_conf["sessions_completed_min"],
                        max_value=self.config.demand_conf["sessions_completed_max"],
                        value=2,
                    )
                    demand_submit = st.form_submit_button("Predict Demand")
                    if demand_submit:
                        input_df = pd.DataFrame(
                            [[age, sessions_completed]], columns=demand_features
                        )
                        X_scaled = scaler.transform(input_df)
                        proba = logreg.predict_proba(X_scaled)[0, 1]
                        pred = logreg.predict(X_scaled)[0]
                        st.success(
                            f"Predicted high demand: {'Yes' if pred else 'No'} (probability: {proba:.2f})"
                        )

    def show_data_tab(self, tab, data_path, meta_path, label):
        with tab:
            if not Path(data_path).exists():
                st.error(f"{label} data file not found.")
            else:
                df = self.loader.load(data_path)
                st.subheader(f"{label} Data")
                st.dataframe(df)
                st.markdown(
                    f"**Total records: {len(df)}** | *Columns: {', '.join(df.columns)}*"
                )
                st.markdown("---")
                # st.subheader(f"{label} Data Privacy Metadata")
                self.display_markdown_file(meta_path)

    @staticmethod
    def display_markdown_file(md_path):
        if Path(md_path).exists():
            with open(md_path, "r") as f:
                st.markdown(f.read())
        else:
            st.warning(f"Metadata file not found: {md_path}")


# --- Run the App ---
if __name__ == "__main__":
    app = CommunityCareApp()
    app.run()
