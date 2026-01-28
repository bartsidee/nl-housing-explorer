# Data Sources & Attribution

Data van Nederlandse overheid bronnen.

## 📊 Primary Data Sources

### 1. CBS - Kerncijfers Wijken en Buurten
- **Source:** https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2024
- **License:** CC BY 4.0
- **Coverage:** 2020-2025
- **Data:** Demographics, income, housing, education

### 2. CBS - Nabijheid Voorzieningen (D-86134NED)
- **Source:** https://opendata.cbs.nl/statline/
- **License:** CC BY 4.0
- **Data:** Distance to services (train, highways, libraries)
- **Year:** 2024

### 3. CBS - SES Scores (85900NED)
- **Source:** https://opendata.cbs.nl/statline/
- **License:** CC BY 4.0
- **Data:** Socio-economic status per neighborhood

### 4. Politie Open Data
- **Source:** https://data.politie.nl
- **License:** CC0 (Public Domain)
- **Coverage:** 2020-2025
- **Data:** Crime rates, burglary statistics

### 5. PDOK - Geographic Boundaries
- **Source:** https://www.pdok.nl
- **License:** CC0 (Public Domain)
- **Data:** Municipality/district/neighborhood boundaries
- **Format:** GeoJSON via WFS API

### 6. PDOK - Elevation Data (AHN)
- **Source:** https://www.pdok.nl/ahn
- **License:** CC0 (Public Domain)
- **Data:** Average elevation (NAP) per area

### 7. RIVM - Green Space
- **Source:** RIVM WFS Service
- **License:** CC BY 4.0
- **Data:** Green space percentage per neighborhood
- **Year:** 2022

---

## 📥 Data Download

**📖 Zie [DATA_DOWNLOAD_GUIDE.md](DATA_DOWNLOAD_GUIDE.md) voor details**

| Dataset | Method | Required |
|---------|--------|----------|
| CBS KWB | Manual | ✅ Yes |
| CBS Nabijheid | Manual | ✅ Yes |
| CBS SES | Manual | ✅ Yes |
| Crime Data | Script | ✅ Yes |
| PDOK Geo | Auto | ⚡ Automatic |
| RIVM Groen | Auto | ⚡ Automatic |
| NAP Heights | Included | ✅ Included |

Quick start:
```bash
python scripts/download_veiligheid_data.py  # Crime data
# + Manual CBS downloads (zie guide)
```

---

## 🔄 Data Pipeline

```
Raw Data → process_multiyear_trends.py → processed/current/main_data_with_trends.parquet → app.py
```

---

## 📜 License Summary

All data sources are **open data**:
- **CC BY 4.0**: Attribution required
- **CC0**: Public domain

Dashboard complies with all license requirements.

---

## 🙏 Attribution

Data by:
- CBS (Centraal Bureau voor de Statistiek)
- Politie Nederland
- PDOK (Publieke Dienstverlening Op de Kaart)
- RIVM (Rijksinstituut voor Volksgezondheid en Milieu)

---

**Last Updated:** January 2026
