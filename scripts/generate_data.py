from pathlib import Path
import pandas as pd
import tomllib
import uuid
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

with open("conf/config.toml", "rb") as f:
    CONFIG = tomllib.load(f)

seg_conf = CONFIG["segmentation"]
demand_conf = CONFIG["demand"]
ds = CONFIG["data_sources"]


def generate_clients(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic client data."""
    clients = []
    for _ in range(n):
        client_id = str(uuid.uuid4())
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
    services = []
    for _ in range(n):
        client_id = str(uuid.uuid4())
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


def main() -> None:
    clients = generate_clients()
    services = generate_services()
    Path(ds["client_data_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(ds["service_data_path"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clients).to_csv(ds["client_data_path"], index=False)
    pd.DataFrame(services).to_csv(ds["service_data_path"], index=False)
    print("Data generated.")


if __name__ == "__main__":
    main()
