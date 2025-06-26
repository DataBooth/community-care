import duckdb
import pandas as pd
from faker import Faker
import numpy as np

fake = Faker()


def make_sample(n=100):
    data = []
    for _ in range(n):
        data.append(
            {
                "age": np.random.randint(18, 80),
                "service_count": np.random.randint(1, 10),
                "gender": np.random.choice(["Male", "Female"]),
                "income_bracket": np.random.choice(["low", "medium", "high"]),
                "service_type": np.random.choice(
                    ["Health", "Employment", "Food", "Housing"]
                ),
                "presenting_issue": np.random.choice(
                    ["Health", "Unemployment", "Mental Health", "Homelessness"]
                ),
                "y": np.random.randint(0, 2),
            }
        )
    return pd.DataFrame(data)


if __name__ == "__main__":
    df = make_sample(100)
    con = duckdb.connect("data/sample_data.duckdb")
    con.execute("CREATE OR REPLACE TABLE clients AS SELECT * FROM df")
    con.close()
    print("Sample data written to DuckDB (sample_data.duckdb, table: clients)")
