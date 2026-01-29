# TLO Data Pipeline

A structured, reproducible data-pipeline framework for preparing, validating, and reporting resource files for the **Thanzi-La-Onse (TLO) Model**.

This repository is designed to support multi-country data ingestion, enforce data quality through explicit runtime checks, and produce standardized resource files and reports for downstream TLO model analyses.

---

## Key Principles

- **Reproducibility**  
  Clear separation of inputs, code, and generated outputs ensures that results can be regenerated consistently.

- **Validation-first design**  
  Built-in runtime checks flag data issues early (e.g. invalid age ranges, negative counts, unexpected categories).

- **Framework, not scripts**  
  Core logic lives in an importable Python package, enabling testing, reuse, and future interfaces (CLI / UI).

- **Country-aware pipelines**  
  Configuration is YAML-driven, with country-specific overrides to avoid hard-coding logic.

---

## Repository Structure

```text
TLOdata_pipeline/
├── src/
│   └── tlo_data_pipeline/
│       ├── config.py        # Configuration loading and merging
│       ├── paths.py         # Canonical input/output path resolution
│       └── demography/
│           ├── census.py
│           ├── dhs.py
│           ├── wpp.py
│           ├── checks.py    # Runtime data validation checks
│           └── reporting/
│               └── build_report.py
│
├── config/
│   ├── pipeline_setup.yaml
│   └── countries/
│       ├── mw.yaml
│       └── tz.yaml
│
├── inputs/                  # Raw, external data (not committed)
│   └── demography/
│       ├── census/
│       ├── dhs/
│       └── wpp/
│
├── outputs/                 # Generated resources and reports
│   ├── resources/
│   └── reports/
│
├── tests/                   # Unit and contract tests
├── docs/                    # Documentation (design notes, usage)
├── pyproject.toml
└── README.md

