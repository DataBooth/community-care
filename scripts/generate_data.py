import os
import random
import tomllib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import pandas as pd

# --- Load config ---
with open("conf/config.toml", "rb") as f:
    CONFIG = tomllib.load(f)

seg_conf = CONFIG["segmentation"]
demand_conf = CONFIG["demand"]
ds = CONFIG["data_sources"]


def get_parquet_key():
    try:
        import streamlit as st

        return st.secrets["encryption"]["parquet_key"]
    except Exception:
        return os.environ["PARQUET_KEY"]


def generate_friendly_client_ids(n: int) -> List[str]:
    """Generate unique, friendly client IDs with reversed year, random increments, and short UUID."""
    year = datetime.now().year
    reversed_year = str(year)[::-1]
    client_ids = []
    seq_num = 1
    for _ in range(n):
        short_uuid = uuid.uuid4().hex[:6]
        seq_str = f"{seq_num:05d}"  # zero-padded to 5 digits
        client_id = f"CC-{reversed_year}-{seq_str}-{short_uuid}"
        client_ids.append(client_id)
        seq_num += random.randint(1, 9)
    return client_ids


def generate_clients(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic client data."""
    client_ids = generate_friendly_client_ids(n)
    clients = []
    for client_id in client_ids:
        age = random.randint(seg_conf["age_min"], seg_conf["age_max"])
        gender = random.choice(seg_conf["genders"])
        postcode = str(
            random.randint(seg_conf["postcode_min"], seg_conf["postcode_max"])
        )
        income = random.choice(seg_conf["income_brackets"])
        service_type = random.choice(seg_conf["service_types"])
        service_count = random.randint(
            seg_conf["service_count_min"], seg_conf["service_count_max"]
        )
        last_service_date = (
            datetime.now()
            - timedelta(days=random.randint(1, seg_conf["last_service_days_back"]))
        ).strftime("%Y-%m-%d")
        presenting_issue = random.choice(seg_conf["presenting_issues"])
        clients.append(
            {
                "client_id": client_id,
                "age": age,
                "gender": gender,
                "postcode": postcode,
                "income_bracket": income,
                "service_type": service_type,
                "service_count": service_count,
                "last_service_date": last_service_date,
                "presenting_issue": presenting_issue,
            }
        )
    return clients


def generate_services(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic service demand data."""
    client_ids = generate_friendly_client_ids(n)
    services = []
    for client_id in client_ids:
        service_type = random.choice(demand_conf["service_types"])
        referral_source = random.choice(demand_conf["referral_sources"])
        age = random.randint(demand_conf["age_min"], demand_conf["age_max"])
        presenting_issue = random.choice(demand_conf["presenting_issues"])
        sessions_completed = random.randint(
            demand_conf["sessions_completed_min"], demand_conf["sessions_completed_max"]
        )
        sessions_needed = sessions_completed + random.randint(
            0, demand_conf["sessions_needed_extra_max"]
        )
        registration_date = (
            datetime.now()
            - timedelta(days=random.randint(1, demand_conf["registration_days_back"]))
        ).strftime("%Y-%m-%d")
        outcome_status = random.choice(demand_conf["outcome_statuses"])
        services.append(
            {
                "client_id": client_id,
                "service_type": service_type,
                "referral_source": referral_source,
                "age": age,
                "presenting_issue": presenting_issue,
                "sessions_completed": sessions_completed,
                "sessions_needed": sessions_needed,
                "registration_date": registration_date,
                "outcome_status": outcome_status,
            }
        )
    return services


def write_encrypted_parquet(df: pd.DataFrame, out_path: str, parquet_key: str):
    con = duckdb.connect(database=":memory:")
    con.register("tbl", df)
    con.execute(f"PRAGMA add_parquet_key('footer_key', '{parquet_key}');")
    con.execute(f"""
        COPY tbl TO '{out_path}'
        (ENCRYPTION_CONFIG {{footer_key: 'footer_key'}})
    """)
    con.unregister("tbl")
    con.close()
    print(f"Encrypted Parquet file written to {out_path}")


def main() -> None:
    clients = generate_clients(seg_conf["n_record"])
    services = generate_services(demand_conf["n_record"])
    Path(ds["client_data_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(ds["service_data_path"]).parent.mkdir(parents=True, exist_ok=True)
    parquet_key = get_parquet_key()
    write_encrypted_parquet(pd.DataFrame(clients), ds["client_data_path"], parquet_key)
    write_encrypted_parquet(
        pd.DataFrame(services),
        ds["service_data_path"],
        parquet_key,
    )
    print("Data generated and encrypted.")


if __name__ == "__main__":
    main()
