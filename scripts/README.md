# 📜 Scripts Overview

## 🎯 Core Scripts

### 1. `scripts/process_multiyear_trends.py` **Main Import Script**

**Purpose:** Process all KWB data, calculate trends, generate dashboard data

**Usage:**
```bash
# Normal run
python scripts/process_multiyear_trends.py

# Force regenerate all files
python scripts/process_multiyear_trends.py --force

# Process specific years
python scripts/process_multiyear_trends.py --years 2024 2025

# See all options
python scripts/process_multiyear_trends.py --help
```

**Arguments:**
- `--force, -f` - Force reprocess all years (regenerate files)
- `--years, -y` - Specific years to process (default: 2020-2025)
- `--no-veiligheid` - Skip crime data integration

**Output:**
- `data/processed/{year}/main_data.parquet` - Year-specific data
- `data/processed/current/main_data_with_trends.parquet` - Final dashboard data
- `data/processed/current/metadata.json` - Processing metadata

**When to use:**
- After downloading new source data
- To regenerate processed files
- After updating processing logic
- Regular data updates

**Documentation:** See [PROCESSING_GUIDE.md](PROCESSING_GUIDE.md)

---

### 2. `scripts/download_veiligheid_data.py` **Crime Data**

**Purpose:** Download crime/safety data from Politie Nederland

**Usage:**
```bash
python scripts/download_veiligheid_data.py
```

**Output:**
- `data/raw/veiligheid/{year}/politie_misdrijven_{year}.csv`

**When to use:**
- Initial setup / New year data becomes available

**Alternative:** Manual download from https://data.politie.nl

---

### 3. `scripts/download_nap_data_fast.py` **NAP Elevation** 

**Purpose:** Download elevation data from PDOK AHN service

**Usage:**
```bash
python scripts/download_nap_data_fast.py --level all --workers 8
```

**Output:**
- `data/geo/cache/ahn/*.csv`

**When to use:**
- Initial setup

