# 🛠️ Setup Instructions

Guide to set up NL Location Insights locally or for deployment.

## 📋 Prerequisites

- Python 3.9+
- pip
- Git
- ~500 MB disk space

## 🚀 Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/bartsidee/nl-location-insights.git
cd nl-location-insights
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Raw Data

**📥 Zie [DATA_DOWNLOAD_GUIDE.md](DATA_DOWNLOAD_GUIDE.md) voor details**

Quick download:

| Dataset | Methode |
|---------|---------|
| CBS KWB | Manual download ([CBS Website](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2024)) |
| Crime Data | `python scripts/download_veiligheid_data.py` |
| CBS Nabijheid | Manual ([CBS Open Data](https://opendata.cbs.nl/statline/), zoek: 86134NED) |
| CBS SES | Manual ([CBS Open Data](https://opendata.cbs.nl/statline/), zoek: 85900NED) |

Verify downloads:
```bash
python -c "
from pathlib import Path
for y in range(2020, 2026):
    kwb = Path(f'data/raw/kwb/{y}/kwb-{y}.xlsx').exists()
    crime = Path(f'data/raw/veiligheid/{y}/politie_misdrijven_{y}.csv').exists()
    print(f'{y}: KWB={kwb}, Crime={crime}')
"
```

### 5. Process Data

```bash
# Normal processing (skip existing years)
python scripts/process_multiyear_trends.py

# Or force regenerate all files
python scripts/process_multiyear_trends.py --force
```

This generates `data/processed/current/main_data_with_trends.parquet` (~5-10 min)

**See [PROCESSING_GUIDE.md](PROCESSING_GUIDE.md) for advanced usage**

### 6. Run Dashboard

```bash
streamlit run app.py
```

Dashboard opens at http://localhost:8501

---

## 🌐 Deployment (Streamlit Cloud)

**📊 Data Strategy:** Zie [DATA_GIT_STRATEGY.md](DATA_GIT_STRATEGY.md)

### 1. Prepare Data

```bash
# Generate processed data
python scripts/process_multiyear_trends.py

# Verify
ls -lh data/processed/current/main_data_with_trends.parquet
```

These files are included in git:
- ✅ `data/processed/current/` - Processed data
- ✅ `data/geo/cache/ahn/` - NAP heights  
- ✅ `data/presets/` - Example configs

### 2. Push to GitHub

```bash
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select repository: `bartsidee/nl-location-insights`
4. Main file: `app.py`
5. Click "Deploy"

Wait ~2-3 minutes.

### 4. Verify Deployment

Test:
- [ ] All tabs load
- [ ] Maps display
- [ ] Custom scores work
- [ ] URL sharing works

---

## 🔧 Troubleshooting

### Module not found

```bash
pip install --upgrade -r requirements.txt
```

### Data file not found

```bash
python scripts/process_multiyear_trends.py
ls data/processed/current/main_data_with_trends.parquet
```

### Missing raw data

```bash
ls data/raw/kwb/2025/kwb-2025.xlsx
ls data/raw/veiligheid/2025/politie_misdrijven_2025.csv
ls data/raw/nabijheid/2024/Observations.csv
ls data/raw/ses/85900NED/Observations.csv
```

### Streamlit won't start

```bash
# Kill existing process
lsof -ti:8501 | xargs kill -9  # macOS/Linux

# Try different port
streamlit run app.py --server.port 8502
```

### Maps not displaying

```bash
pip install folium>=0.15.0 streamlit-folium>=0.15.0 geopandas>=0.14.0
```

---

## 📊 Update Data

To update with latest CBS data:

```bash
# 1. Download new raw data (see Step 4)
# 2. Re-process
python scripts/process_multiyear_trends.py
# 3. Restart dashboard
streamlit run app.py
```

---

## 🔐 Configuration

### Streamlit Config

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"

[server]
port = 8501
```

---

## 🧪 Development Mode

Auto-reload on save:

```bash
streamlit run app.py --server.runOnSave=true
```

---

## 🆘 Getting Help

1. Check [README.md](README.md)
2. Review [DATA_SOURCES.md](DATA_SOURCES.md)
3. Open GitHub issue
4. Streamlit docs: https://docs.streamlit.io

---

## ✅ Setup Complete!

- ✅ Dependencies installed
- ✅ Raw data downloaded
- ✅ Processed data generated
- ✅ Dashboard running

**Next:** Explore at http://localhost:8501
