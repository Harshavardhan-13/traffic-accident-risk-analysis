# Source data and local setup

Download **Traffic Violations** from the [Data.gov catalog](https://catalog.data.gov/dataset/traffic-violations), published by [Montgomery County, Maryland](https://data.montgomerycountymd.gov/d/4mse-ku6q). Choose its CSV resource and save the download as `Traffic_Violations.csv` in the repository root.

The source covers electronic violations issued in Montgomery County. It is not a census of all Maryland drivers, all trips, or all crashes. The online dataset changes; the evaluated local snapshot covers January 1, 2012 through September 16, 2024. See `reports/data_audit.json` for the evaluated file's size, hash, row counts, and date bounds.

```text
traffic-accident-risk-analysis/
|-- Traffic_Violations.csv          # downloaded input, ignored by Git
|-- data/
|   |-- README.md
|   `-- traffic_prepared.csv        # generated input, ignored by Git
|-- notebooks/
|-- scripts/
|-- src/
`-- reports/                        # aggregate audit and metrics only
```

The original `Project-1 Traffic violation.xlsx` and `cleaned_traffic_violations.csv` are retained locally but are not required by the revised workflow. The Excel-based coursework used a truncated subset, and the historical cleaned CSV is not substituted for the full raw source. `Book2.csv` and `Book2.xlsx` are also not pipeline inputs.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_analysis.py --prepare-only
```

To use a CSV stored elsewhere:

```bash
python scripts/run_analysis.py --raw '/absolute/path/Traffic_Violations.csv' --prepare-only
```

Preparation reads in chunks, audits exclusions, and writes local row-level data to `data/traffic_prepared.csv`. Schema changes stop the pipeline with an explicit error rather than silently selecting unrelated columns. Fresh downloads need not reproduce the older snapshot's totals or model scores.

All source downloads and generated row-level data are excluded from Git. `reports/*.csv` are small aggregate metric tables and are intentionally included. Download from the original publisher and follow its applicable terms.
