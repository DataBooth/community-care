import tomllib
from typing import Any, Dict, Optional

import duckdb
import numpy as np
import pandas as pd
from synthpop import CARTMethod, DataProcessor, MissingDataHandler


class CommunityCareSynthDataGenerator:
    """
    Generates a synthetic dataset that mimics the demographic and social characteristics
    of CommunityCare's older women client base using the python-synthpop library.
    Provides methods to persist data to DuckDB and run SQL queries.
    """

    def __init__(self, config_path: str = "conf/config.toml"):
        """
        Initialize the generator using parameters from a TOML config file.

        Args:
            config_path (str): Path to TOML configuration file.
        """
        self.config = self._load_config(config_path)
        self.n_samples = int(self.config["n_samples"])
        self.random_state = int(self.config["random_state"])
        self.rng = np.random.default_rng(self.random_state)
        self.db_path = self.config.get("duckdb_path", "data/communitycare_synth.db")
        self.table_name = self.config.get("duckdb_table", "synthetic_clients")
        self.metadata: Dict[str, str] = {}
        self.original_df: pd.DataFrame = pd.DataFrame()
        self.synthetic_df: pd.DataFrame = pd.DataFrame()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the TOML configuration file and parses the synthetic_pop_params section.

        Args:
            config_path (str): Path to config file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        params = config.get("synthetic_pop_params", {})
        # Parse string lists into Python lists
        for key in params:
            if key.endswith("_labels") or key.endswith("_probs"):
                params[key] = [
                    x.strip() if "_labels" in key else float(x.strip())
                    for x in params[key].split(",")
                ]
        return params

    def create_seed_data(self) -> pd.DataFrame:
        """
        Create a seed DataFrame with the desired marginal distributions from config.

        Returns:
            pd.DataFrame: The seed dataset.
        """
        gender = pd.Series(
            pd.Categorical(
                self.rng.choice(
                    self.config["gender_labels"],
                    size=self.n_samples,
                    p=self.config["gender_probs"],
                ),
                categories=self.config["gender_labels"],
            ),
            name="gender",
        )
        age_group = pd.Series(
            pd.Categorical(
                self.rng.choice(
                    self.config["age_group_labels"],
                    size=self.n_samples,
                    p=self.config["age_group_probs"],
                ),
                categories=self.config["age_group_labels"],
            ),
            name="age_group",
        )
        indigenous = pd.Series(
            pd.Categorical(
                self.rng.choice(
                    self.config["indigenous_labels"],
                    size=self.n_samples,
                    p=self.config["indigenous_probs"],
                ),
                categories=self.config["indigenous_labels"],
            ),
            name="indigenous",
        )
        marital = pd.Series(
            pd.Categorical(
                self.rng.choice(
                    self.config["marital_labels"],
                    size=self.n_samples,
                    p=self.config["marital_probs"],
                ),
                categories=self.config["marital_labels"],
            ),
            name="marital",
        )
        housing = pd.Series(
            pd.Categorical(
                self.rng.choice(
                    self.config["housing_labels"],
                    size=self.n_samples,
                    p=self.config["housing_probs"],
                ),
                categories=self.config["housing_labels"],
            ),
            name="housing",
        )
        df = pd.concat([gender, age_group, indigenous, marital, housing], axis=1)
        return df

    def fit(self) -> None:
        """
        Fit the synthpop pipeline to the seed data and generate synthetic data.
        """
        self.original_df = self.create_seed_data()
        md_handler = MissingDataHandler()
        self.metadata = md_handler.get_column_dtypes(self.original_df)
        missingness_dict = md_handler.detect_missingness(self.original_df)
        imputed_df = md_handler.apply_imputation(self.original_df, missingness_dict)
        processor = DataProcessor(self.metadata)
        processed_data = processor.preprocess(imputed_df)
        cart = CARTMethod(
            self.metadata,
            smoothing=True,
            proper=True,
            minibucket=5,
            random_state=self.random_state,
        )
        cart.fit(processed_data)
        synthetic_processed = cart.sample(self.n_samples)
        self.synthetic_df = processor.postprocess(synthetic_processed)

    def get_synthetic_data(self) -> pd.DataFrame:
        """
        Returns the generated synthetic DataFrame.

        Returns:
            pd.DataFrame: Synthetic dataset.
        """
        if self.synthetic_df.empty:
            self.fit()
        return self.synthetic_df

    def get_original_data(self) -> pd.DataFrame:
        """
        Returns the seed/original DataFrame.

        Returns:
            pd.DataFrame: Original dataset.
        """
        if self.original_df.empty:
            self.fit()
        return self.original_df

    def save_to_duckdb(
        self, db_path: Optional[str] = None, table_name: Optional[str] = None
    ) -> None:
        """
        Saves the synthetic dataset to a DuckDB database.

        Args:
            db_path (str): Path to the DuckDB database file.
            table_name (str): Table name to use in DuckDB.
        """
        db_path = db_path or self.db_path
        table_name = table_name or self.table_name
        df = self.get_synthetic_data()
        con = duckdb.connect(db_path)
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        con.close()

    def run_sql_query(self, query: str, db_path: Optional[str] = None) -> pd.DataFrame:
        """
        Runs a SQL query on the DuckDB database and returns the result as a DataFrame.

        Args:
            query (str): SQL query to execute.
            db_path (str, optional): Path to the DuckDB database file. Uses last used if not provided.

        Returns:
            pd.DataFrame: Query result.
        """
        db = db_path or self.db_path
        if db is None:
            raise ValueError("No DuckDB database path specified.")
        con = duckdb.connect(db)
        result = con.execute(query).df()
        con.close()
        return result

    def compare_category_percentages(
        self, df: pd.DataFrame = None, specs: dict = None
    ) -> pd.DataFrame:
        """
        Compare observed percentages in df to specified probabilities from specs.

        Args:
            df (pd.DataFrame, optional): DataFrame to compare. Defaults to synthetic_df.
            specs (dict, optional): Spec dict (labels/probs). Defaults to config-parsed specs.

        Returns:
            pd.DataFrame: Summary table with Variable, Category, Specified %, Observed %, Difference %.
        """
        if df is None:
            df = self.get_synthetic_data()
        if specs is None:
            # Build specs from self.config (parsed in __init__)
            specs = {}
            for var in ["gender", "age_group", "indigenous", "marital", "housing"]:
                specs[var] = {
                    "labels": self.config[f"{var}_labels"],
                    "probs": self.config[f"{var}_probs"],
                }
        results = []
        for col, val in specs.items():
            labels = val["labels"]
            probs = val["probs"]
            obs = df[col].value_counts(normalize=True).reindex(labels, fill_value=0)
            for i, label in enumerate(labels):
                results.append(
                    {
                        "Variable": col,
                        "Category": label,
                        "Specified %": round(100 * probs[i], 2),
                        "Observed %": round(100 * obs[label], 2),
                        "Difference %": round(100 * (obs[label] - probs[i]), 2),
                    }
                )
        return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    generator = CommunityCareSynthDataGenerator(config_path="conf/config.toml")
    synthetic_df = generator.get_synthetic_data()
    print("Sample synthetic data:")
    print(synthetic_df.head())

    # Save to DuckDB (uses config values)
    generator.save_to_duckdb()
    print(f"Synthetic data written to DuckDB at {generator.db_path}\n")

    # Run a SQL query
    query = """
        SELECT gender, age_group, COUNT(*) as count
        FROM synthetic_clients
        GROUP BY gender, age_group
        ORDER BY gender, age_group
    """
    result_df = generator.run_sql_query(query)
    print("SQL query result:")
    print(f"{result_df}\n")

    total_clients = generator.run_sql_query("SELECT COUNT(*) FROM synthetic_clients")
    print(f"Total clients in synthetic dataset: {total_clients.iloc[0, 0]}\n")

    # Compare observed to specified percentages
    summary = generator.compare_category_percentages()
    print(summary)
