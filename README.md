# TLO Data Pipeline

A structured, reproducible data-pipeline framework for preparing resource files for the **Thanzi-La-Onse (TLO) Model**.

This repository is designed to support multi-country data ingestion, enforce data quality through explicit runtime checks, and produce standardized resource files for the TLO model.

# Installation
```
# Clone the repository
git clone https://github.com/TLOmodel/TLOdata_pipeline.git

# Navigate to the repository
cd TLOdata_pipeline

# Create a virtual environment and install the requirements
python3.11 -m venv pipeline_env
source pipeline_env/bin/activate
pip install -r requirements/dev.txt
pip install -e .
```