# Data Processed Directory Structure

**Laatste update:** 2026-01-25  
**Status:** Production ready met multi-year trends

---

## 📁 Directory Structuur

```
data/processed/
├── 2020/
│   └── main_data.parquet      # KWB 2020 + SES 2020 (volledig geprocessed)
├── 2021/
│   └── main_data.parquet      # KWB 2021 + SES 2021 (volledig geprocessed)
├── 2022/
│   └── main_data.parquet      # KWB 2022 + SES 2022 (volledig geprocessed)
├── 2023/
│   └── main_data.parquet      # KWB 2023 + SES 2022 (volledig geprocessed)
└── current/
    ├── main_data_with_trends.parquet  # ⭐ GEBRUIKT DOOR DASHBOARD
    └── metadata.json                  # Trend metadata
```

---

## 🎯 Welke File Gebruikt Wat?

### Dashboard (app.py)
```python
# Laadt:
data/processed/current/main_data_with_trends.parquet  # Met trends (ENIGE OPTIE)
```

### Trend Processing (scripts/process_multiyear_trends.py)
```python
# Input (genereert indien niet aanwezig):
data/processed/2020/main_data.parquet
data/processed/2021/main_data.parquet
data/processed/2022/main_data.parquet
data/processed/2023/main_data.parquet

# Output:
data/processed/current/main_data_with_trends.parquet
data/processed/current/metadata.json
```

---

## 📊 File Details

### Multi-Year Data (2020-2023)
**Doel:** Bron voor trend berekeningen

**Structuur:**
- KWB data voor dat jaar (CBS Kerncijfers Wijken en Buurten)
- SES data voor dat jaar (2014-2022 beschikbaar)
- Afgeleide metrics (pct_children, area_per_person_m2, etc.)
- RIVM Groen percentage (2022 data, gebruikt voor alle jaren)

**Columns:** ~56 indicatoren
**Rows:** 15,708 - 18,116 (groeit per jaar)

### Current Data
**Doel:** Actieve dataset voor dashboard

#### `main_data_with_trends.parquet` ⭐
- **ENIGE DATASET VOOR DASHBOARD**
- 2023 data + 34 trend kolommen
- Trends: 2020 → 2023 (3 jaar)
- Threshold: 2% per jaar × 3 = 6% totaal
- 11 indicatoren met trends:
  - pct_children
  - pct_families
  - bev_dich
  - ses_overall
  - ses_onderwijs
  - g_ink_pi
  - p_arb_pp
  - groen_percentage
  - p_koopw
  - g_hhgro
  - p_ink_li
  - area_per_person_m2

**Trend Kolommen per indicator:**
- `trend_{indicator}_abs` - Absolute verandering
- `trend_{indicator}_pct` - Percentage verandering
- `trend_{indicator}_dir` - Richting (stijgend/dalend/stabiel)
- `trend_score` - Composite gewogen trend score

#### `metadata.json`
```json
{
  "generated_at": "2026-01-25T...",
  "base_year": 2023,
  "trend_years": "2020-2023",
  "trend_period_years": 3,
  "rows": 18116,
  "columns": 90,
  "trend_columns": 34,
  "note": "Real trends calculated from actual multi-year CBS KWB data (2020-2023)",
  "threshold_pct_per_year": 2.0
}
```

---

## 🔄 Data Pipeline

### Stap 1: Multi-Year Processing
```bash
python3 scripts/process_multiyear_trends.py
```

**Wat gebeurt er:**
1. Check of year data bestaat in `data/processed/{YEAR}/`
2. Zo niet, process KWB + SES data voor dat jaar:
   - Load `data/raw/kwb/{YEAR}/kwb-{YEAR}.xlsx`
   - Load `data/raw/ses/85900NED/` (filter op jaar)
   - Merge, clean, calculate derived metrics
   - Save naar `data/processed/{YEAR}/main_data.parquet`
3. Calculate trends tussen eerste en laatste jaar
4. Merge trends met current data
5. Save `current/main_data_with_trends.parquet`

### Stap 2: Dashboard
```bash
streamlit run app.py
```

**Wat gebeurt er:**
1. Check `data/processed/current/main_data_with_trends.parquet`
2. Load data (automatisch cached door Streamlit)
3. Toon trends in UI

---

## ⚠️ NIET VERWIJDEREN

**Bewaar altijd:**
- `2020/`, `2021/`, `2022/`, `2023/` folders
  → Nodig voor herberekening trends
- `current/main_data_with_trends.parquet`
  → Dashboard dataset (enige benodigde file!)
- `current/metadata.json`
  → Trend informatie

**Mag hergenerated worden:**
- Alle files kunnen opnieuw gegenereerd worden via:
  ```bash
  python3 scripts/process_multiyear_trends.py --force
  ```

---

## 🗑️ Opruiming Historie

**2026-01-25 (2e cleanup):**
Verwijderd (redundante file):
- `current/main_data.csv` (7.9 MB) - Dubbel, trends file is leidend

**2026-01-27 (Final cleanup):**
Verwijderd (oude naming convention / intermediate files):
- `kwb_2023_cleaned.csv` (3.5 MB)
- `kwb_2023_scored.csv` (5.6 MB) - Oud scoring systeem
- `kwb_2023_with_ses.csv` (8.6 MB) - Legacy archive format
- `ses/ses_2022.csv` - Intermediate file
- `ses/ses_2023.csv` - Intermediate file

**2026-01-28 (Parquet migration):**
Switched from CSV to Parquet format:
- `*.csv` → `*.parquet` (all processed data)
- 50% smaller file sizes with gzip compression
- Faster loading times

**Reden:** Vervangen door `current/main_data_with_trends.parquet` (production data)

---

## 📈 Trend Analysis Details

### Classificatie
- **Stijgend ↑**: > +6% over 3 jaar (+2% per jaar)
- **Stabiel ↔**: -6% tot +6% over 3 jaar
- **Dalend ↓**: < -6% over 3 jaar (-2% per jaar)

### Coverage
- **Trend Score**: 18,116 locaties (100%)
- **SES Trends**: 11,577 locaties (63.9%)
- **Overige Trends**: 15,708 - 18,116 locaties (86-100%)

### Distributies (2020-2023)
```
Overall Trend:     ↑14%  ↔70%  ↓16%
Kinderen:          ↑18%  ↔48%  ↓34%  (dalende trend NL-breed)
SES Overall:       ↑ 1%  ↔98%  ↓ 1%  (zeer stabiel)
Bevolkingsdichtheid: ↑20%  ↔71%  ↓ 9%  (urbanisatie)
```

---

## 🚀 Regenerate Data

### Volledige herberekening:
```bash
# Verwijder oude processed data
rm -rf data/processed/2020 data/processed/2021 data/processed/2022

# Herbereken alles
python3 scripts/process_multiyear_trends.py

# Start dashboard
streamlit run app.py
```

### Alleen trends herberekenen (sneller):
```bash
# Year data blijft bestaan, alleen trends worden herberekend
python3 scripts/process_multiyear_trends.py
```

---

## 📊 File Sizes (indicatief)

```
2020/main_data.parquet              ~3.5 MB (gzip compressed)
2021/main_data.parquet              ~3.5 MB (gzip compressed)
2022/main_data.parquet              ~3.5 MB (gzip compressed)
2023/main_data.parquet              ~3.6 MB (gzip compressed)
current/main_data_with_trends.parquet ~7 MB  (56 base + 34 trend columns, gzip compressed)
current/metadata.json               <1 KB
```

**Totaal:** ~44 MB

---

## ✅ Quality Checks

### Verify Trends
```bash
python3 scripts/verify_trend_data.py
```

Output:
- ✅ Metadata aanwezig
- ✅ Trend kolommen complete
- ✅ Data diversiteit (>15,000 unieke trend scores)
- ✅ Source files (2020-2023) aanwezig

### Check Coverage
```python
import pandas as pd

df = pd.read_parquet('data/processed/current/main_data_with_trends.parquet')

# Check trend columns
trend_cols = [c for c in df.columns if c.startswith('trend_')]
print(f"Trend columns: {len(trend_cols)}")

# Check coverage
for col in ['trend_score', 'trend_ses_overall_pct', 'trend_pct_children_pct']:
    coverage = df[col].notna().sum() / len(df) * 100
    print(f"{col}: {coverage:.1f}% coverage")
```

---

**Data is production-ready en volledig gedocumenteerd!** 🎯
