# 📥 Data Download Guide

Instructies voor het downloaden van benodigde data sources.

## 📊 Overzicht

| Data Source | Methode | Vereist |
|-------------|---------|---------|
| CBS KWB | Manual | ✅ Ja |
| CBS SES | Manual | ✅ Ja |
| CBS Nabijheid | Manual | ✅ Ja |
| Crime Data | Script | ✅ Ja |
| PDOK Geo | Automatic | ⚡ Auto |
| RIVM Groen | Automatic | ⚡ Auto |
| NAP Heights | Included | ✅ Included |

---

## 1️⃣ CBS Kerncijfers Wijken en Buurten (KWB)

**Vereist:** Ja (primaire dataset)

### Download

1. Ga naar: https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2024
2. Scroll naar "Kerncijfers per wijk en buurt"
3. Download Excel bestanden voor 2020-2025:
   - https://www.cbs.nl/-/media/_excel/2021/42/kwb-2020.xlsx
   - https://www.cbs.nl/-/media/_excel/2022/42/kwb-2021.xlsx
   - https://www.cbs.nl/-/media/_excel/2023/42/kwb-2022.xlsx
   - https://www.cbs.nl/-/media/_excel/2024/42/kwb-2023.xlsx
   - https://www.cbs.nl/-/media/_excel/2025/42/kwb-2024.xlsx
   - https://www.cbs.nl/-/media/_excel/2026/42/kwb-2025.xlsx

### Plaats Bestanden

```bash
mkdir -p data/raw/kwb/{2020,2021,2022,2023,2024,2025}
mv ~/Downloads/kwb-*.xlsx data/raw/kwb/YEAR/
```

Structuur:
```
data/raw/kwb/2020/kwb-2020.xlsx
data/raw/kwb/2021/kwb-2021.xlsx
...
data/raw/kwb/2025/kwb-2025.xlsx
```

---

## 2️⃣ Politie Criminaliteit Data

**Vereist:** Ja (voor veiligheidsscores)

### Optie A: Geautomatiseerd (Aanbevolen)

```bash
python scripts/download_veiligheid_data.py
```

### Optie B: Manual

1. Ga naar: https://data.politie.nl
2. Zoek "Misdrijven per buurt"
3. Download CSV voor elk jaar (2020-2025)
4. Hernoem naar: `politie_misdrijven_{year}.csv`
5. Plaats in: `data/raw/veiligheid/{year}/`

---

## 3️⃣ CBS Nabijheid Voorzieningen

**Vereist:** Ja (voor afstand tot voorzieningen)

### Download

1. Ga naar: https://opendata.cbs.nl/statline/
2. Zoek: **"86134NED"**
3. Download CSV (ZIP bestand)
4. Pak uit:

```bash
cd ~/Downloads
unzip 86134NED.zip -d nabijheid_2024
mkdir -p data/raw/nabijheid/2024
cp nabijheid_2024/* data/raw/nabijheid/2024/
```

Belangrijkste bestand: `Observations.csv`

---

## 4️⃣ CBS SES Scores

**Vereist:** Ja (voor sociaal-economische status)

### Download

1. Ga naar: https://opendata.cbs.nl/statline/
2. Zoek: **"85900NED"**
3. Download CSV (ZIP bestand)
4. Pak uit:

```bash
cd ~/Downloads
unzip 85900NED.zip -d ses_data
mkdir -p data/raw/ses/85900NED
cp ses_data/* data/raw/ses/85900NED/
```

Belangrijkste bestand: `Observations.csv`

---

## 5️⃣ Automatische Data Sources

Deze worden automatisch geladen tijdens data processing:

- **PDOK Geo:** Gemeente/wijk/buurt grenzen (WFS API)
- **RIVM Groen:** Groenpercentage (WFS API)
- **NAP Hoogte:** Pre-calculated, already included in `data/geo/cache/ahn/`

---

## ✅ Verificatie

Check alle data:

```bash
python -c "
import os
from pathlib import Path

checks = {
    'KWB': [f'data/raw/kwb/{y}/kwb-{y}.xlsx' for y in range(2020, 2026)],
    'Crime': [f'data/raw/veiligheid/{y}/politie_misdrijven_{y}.csv' for y in range(2020, 2026)],
    'Nabijheid': ['data/raw/nabijheid/2024/Observations.csv'],
    'SES': ['data/raw/ses/85900NED/Observations.csv'],
}

print('📋 Data Verification\n' + '='*50)
for name, files in checks.items():
    ok = sum(Path(f).exists() for f in files)
    print(f'{"✅" if ok == len(files) else "❌"} {name}: {ok}/{len(files)}')
"
```

Als alle checks ✅ zijn:

```bash
python scripts/process_multiyear_trends.py
```

---

## 🆘 Troubleshooting

**CBS links werken niet?**
→ Ga naar cbs.nl en zoek handmatig naar "Kerncijfers wijken en buurten"

**Script download_veiligheid_data.py faalt?**
→ Download handmatig via data.politie.nl

**ZIP bevat andere bestanden?**
→ Check of `Observations.csv` aanwezig is

**Disk space error?**
→ Minimaal 500 MB vrije ruimte nodig
