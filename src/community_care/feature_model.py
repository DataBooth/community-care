import itertools
import tomllib

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from formulaic import Formula
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate
from tqdm import tqdm


class FormulaicFeatureModelSelector:
    """
    Model selector with MLflow tracking, using formulaic for feature engineering and scikit-learn for modeling.
    Supports regression (linear, tree-based) and classification (logistic, tree-based).
    """

    def __init__(
        self,
        df,
        candidate_features,
        target,
        task="regression",
        model_types=None,
        always_include=None,
        always_exclude=None,
        poly_feature=None,
        poly_degrees=[1, 2, 3],
        max_features=None,
        scoring=None,
        cv=5,
        mlflow_experiment=None,
        mlflow_tracking_uri=None,
    ):
        self.df = df
        self.candidate_features = candidate_features
        self.target = target
        self.task = task
        self.model_types = model_types or (
            ["linear", "rf"] if task == "regression" else ["logistic", "rf"]
        )
        self.always_include = always_include or []
        self.always_exclude = always_exclude or []
        self.poly_feature = poly_feature
        self.poly_degrees = poly_degrees
        self.max_features = max_features or len(candidate_features)
        if scoring:
            self.scoring = scoring
        else:
            self.scoring = (
                ["neg_root_mean_squared_error", "r2"]
                if task == "regression"
                else ["accuracy", "f1"]
            )
        # Set cv to be at most the number of samples
        self.cv = min(cv, len(df))
        self.results = []
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_tracking_uri = mlflow_tracking_uri

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if self.mlflow_experiment:
            mlflow.set_experiment(self.mlflow_experiment)

    def _make_formula(self, features, poly_degree):
        rhs_terms = []
        for f in features:
            if self.poly_feature and f == self.poly_feature and poly_degree > 1:
                rhs_terms.append(f"I({f}**{poly_degree})")
            else:
                rhs_terms.append(f)
        rhs = " + ".join(rhs_terms)
        return f"{self.target} ~ {rhs}"

    def _get_model(self, model_type):
        if self.task == "regression":
            if model_type == "linear":
                return LinearRegression()
            elif model_type == "rf":
                return RandomForestRegressor(random_state=42)
            else:
                raise ValueError(f"Unknown regression model: {model_type}")
        elif self.task == "classification":
            if model_type == "logistic":
                return LogisticRegression(max_iter=1000)
            elif model_type == "rf":
                return RandomForestClassifier(random_state=42)
            else:
                raise ValueError(f"Unknown classification model: {model_type}")
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def search(self):
        features_to_search = [
            f
            for f in self.candidate_features
            if f not in self.always_include and f not in self.always_exclude
        ]
        all_combos = []
        for n in range(1, self.max_features + 1):
            for combo in itertools.combinations(features_to_search, n):
                all_combos.append(list(self.always_include) + list(combo))

        total_runs = len(all_combos) * len(self.poly_degrees) * len(self.model_types)
        with tqdm(total=total_runs, desc="Model Search", unit="run") as pbar:
            for features in all_combos:
                for degree in self.poly_degrees:
                    formula = self._make_formula(features, degree)
                    # Optionally, show current formula in tqdm bar
                    pbar.set_postfix(
                        {
                            "formula": formula[:30] + "..."
                            if len(formula) > 30
                            else formula
                        }
                    )
                    logger.info(f"Evaluating formula: {formula}")
                    y, X = Formula(formula).get_model_matrix(self.df)
                    y = y.values.ravel()  # Always flatten y for sklearn

                    for model_type in self.model_types:
                        model = self._get_model(model_type)
                        cv_results = cross_validate(
                            model,
                            X,
                            y,
                            scoring=self.scoring,
                            cv=self.cv,
                            return_train_score=False,
                        )
                        result = {
                            "features": features,
                            "poly_degree": degree,
                            "model_type": model_type,
                            "formula": formula,
                        }
                        for metric in self.scoring:
                            result[metric] = np.mean(cv_results[f"test_{metric}"])
                        self.results.append(result)

                        # --- MLflow logging ---
                        with mlflow.start_run(nested=True):
                            mlflow.log_param("features", features)
                            mlflow.log_param("poly_degree", degree)
                            mlflow.log_param("model_type", model_type)
                            mlflow.log_param("formula", formula)
                            for metric in self.scoring:
                                mlflow.log_metric(metric, result[metric])
                            # Fit model for logging - convert integers to floats to avoid issues with sklearn models
                            X_float = X.copy()
                            for col in X_float.columns:
                                if pd.api.types.is_integer_dtype(X_float[col]):
                                    X_float[col] = X_float[col].astype(float)

                            model.fit(X_float, y)
                            signature = mlflow.models.infer_signature(
                                X_float, model.predict(X_float)
                            )
                            mlflow.sklearn.log_model(
                                model,
                                name="model",
                                input_example=X_float.iloc[[0]],
                                signature=signature,
                            )
                        pbar.update(1)  # Advance the progress bar

        ascending = False if self.scoring[0] in ["accuracy", "r2", "f1"] else True
        return pd.DataFrame(self.results).sort_values(
            self.scoring[0], ascending=ascending
        )


def main():
    # --- Load data from DuckDB ---
    db_path = "data/sample_data.duckdb"
    table = "clients"
    con = duckdb.connect(db_path)
    # print(f"{con.sql('SHOW TABLES')}\n")
    sample_df = con.execute(f"SELECT * FROM {table}").df()
    con.close()

    print("Loaded DataFrame from DuckDB:\n")
    print(sample_df.head())

    with open("conf/feature_model.toml", "rb") as f:
        config = tomllib.load(f)

    features_cfg = config["features"]
    model_cfg = config["model"]
    mlflow_cfg = config.get("mlflow", {})

    selector = FormulaicFeatureModelSelector(
        df=sample_df,
        candidate_features=features_cfg["candidate"],
        target=features_cfg["target"],
        always_include=features_cfg.get("always_include", []),
        always_exclude=features_cfg.get("always_exclude", []),
        poly_feature=features_cfg.get("poly_feature"),
        poly_degrees=features_cfg.get("poly_degrees", [1]),
        max_features=features_cfg.get("max_features"),
        task=model_cfg.get("task", "regression"),
        model_types=model_cfg.get("model_types", ["linear", "rf"]),
        scoring=model_cfg.get("scoring", ["neg_root_mean_squared_error", "r2"]),
        cv=model_cfg.get("cv", 5),
        mlflow_experiment=mlflow_cfg.get("experiment_name"),
        # mlflow_tracking_uri=mlflow_cfg.get("tracking_uri"),
    )
    results = selector.search()
    print(results)


if __name__ == "__main__":
    main()
