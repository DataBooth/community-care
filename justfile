# Default
default:
    @just --list

# Generate synthetic data
gen-data:
    python scripts/generate_data.py

# Train models and track with MLflow
train-models:
    python scripts/train_models.py

# Clean up generated models
clean-models:
    rm -rf models

view-train:
    mlflow ui --port 5000


# Run the full workflow: data generation then training
all: gen-data train-models

# Run the Streamlit app
app:
    streamlit run app/main.py
