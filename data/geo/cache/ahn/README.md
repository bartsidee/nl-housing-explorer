# AHN NAP Hoogte Data - Pre-calculated Dataset

⚠️ **BELANGRIJK**: Deze CSV files bevatten **pre-calculated** gemiddelde NAP hoogtes.

## Deployment Strategie

### ✅ Aanbevolen: Commit naar Git

Deze files zijn **klein genoeg** (1-2 MB totaal) om naar git te committen:

```bash
git add data/geo/cache/ahn/*.csv
git commit -m "Add pre-calculated NAP heights"
```

**Voordelen:**
- ✅ Geen WCS API calls in productie nodig
- ✅ Geen grote raster data opslag
- ✅ Instant loading (< 1 sec)
- ✅ Werkt offline
- ✅ Reproduceerbaar

### ❌ Niet Nodig: Grote Raster Files

Je hoeft **GEEN** grote GeoTIFF raster files op te slaan:
- ❌ `raster_amsterdam.tif` (honderden MB)
- ❌ Ruwe hoogte data per pixel

Alleen deze compacte CSV files met **gemiddelden**:
- ✅ `gemeente_nap_heights.csv` (~100 KB)
- ✅ `wijk_nap_heights_all.csv` (~300 KB)
- ✅ `buurt_nap_heights_all.csv` (~1 MB)

## Bestanden

Deze folder bevat gecachte gemiddelde NAP hoogtes per gemeente, wijk en buurt.

## Databron

- **AHN4** (Actueel Hoogtebestand Nederland) via PDOK WCS
- **DTM** (Digital Terrain Model) - maaiveld hoogte zonder gebouwen/vegetatie
- Resolutie: 0.5 meter
- API: https://service.pdok.nl/rws/ahn/wcs/v1_0

## Bestanden

- `gemeente_nap_heights.csv` - Gemiddelde NAP hoogte per gemeente (~342 gemeenten)
- `wijk_nap_heights_all.csv` - Gemiddelde NAP hoogte per wijk (~3,300 wijken)
- `buurt_nap_heights_all.csv` - Gemiddelde NAP hoogte per buurt (~14,400 buurten)

## Data Kolommen

- `gwb_code_10`: CBS gebiedscode (GM*, WK*, BU*)
- `region_name`: Naam van het gebied
- `nap_hoogte_gem`: Gemiddelde hoogte in meters t.o.v. NAP (Normaal Amsterdams Peil)

## Interpretatie

### NAP Hoogte Waarden

- **Negatief** (bijv. -2.5m): Onder zeeniveau (kustgebieden, polders)
- **0 tot 5m**: Laagland, kustzone
- **5 tot 25m**: Licht heuvelend
- **25 tot 100m**: Heuvelachtig
- **> 100m**: Heuvels (vooral Zuid-Limburg)

### Toepassingen

1. **Overstromingsrisico**: Lagere gebieden (vooral < 0m) hebben hoger risico
2. **Ligging**: Indicatie of gebied in laagland, polder of hoger gelegen gebied ligt
3. **Klimaatadaptatie**: Relevant voor zeespiegelstijging planning
4. **Drainage**: Lagere gebieden kunnen vochtiger zijn

### Voorbeelden

- **Amsterdam centrum**: ~-2m NAP (onder zeeniveau, beschermd door dijken)
- **Rotterdam**: ~-5m NAP (laagste punt van Nederland)
- **Utrecht**: ~5m NAP (iets hoger, rand rivierengebied)
- **Nijmegen**: ~15m NAP (stuwwal)
- **Vaals (Zuid-Limburg)**: ~200m NAP (hoogste punt Nederland: 322.4m)

## Data Genereren (Eenmalig!)

⚠️ **Je hoeft dit maar ÉÉN KEER te draaien!**

### Optie 1: Gebruik Bestaande Dataset (Aanbevolen)

Als deze CSV files al bestaan in de repo:

```bash
# Niets doen! Gewoon gebruiken:
python scripts/process_multiyear_trends.py
```

### Optie 2: Genereer Nieuwe Dataset

Alleen nodig als:
- Files niet in repo zitten
- CBS heeft boundaries geüpdatet (yearly)
- Je wilt nieuwere AHN data gebruiken

```bash
# One-time generation (45-60 minuten)
python scripts/generate_nap_dataset.py

# OF: Test eerst met kleine sample
python scripts/download_nap_data.py --test

# OF: Per niveau
python scripts/download_nap_data.py --level gemeente  # 5 min
python scripts/download_nap_data.py --level wijk      # 15 min
python scripts/download_nap_data.py --level buurt     # 45 min
```

**Na generatie: Commit naar git!**

```bash
git add data/geo/cache/ahn/*.csv
git commit -m "Add NAP heights dataset"
```

## Technische Details

- De loader gebruikt WCS (Web Coverage Service) om raster hoogte data op te halen
- Voor elk gebied wordt de gemiddelde hoogte berekend over alle pixels binnen de geometrie
- Resolutie wordt dynamisch aangepast (max 512x512 pixels per gebied) om API niet te overbelasten
- Requests worden vertraagd (0.05-0.2 sec tussen requests) om server te ontzien
- Resultaten worden gecached voor hergebruik

## Beperkingen

1. **Gemiddelde waarde**: Binnen een gebied kan de hoogte sterk variëren
2. **Resolutie trade-off**: Lagere resolutie voor snellere downloads
3. **DTM vs DSM**: We gebruiken DTM (terrain) niet DSM (surface), dus gebouwen worden niet meegenomen
4. **Cache datum**: Data wordt gecached, dus recente wijzigingen (dijkverhogingen, etc.) zijn mogelijk niet opgenomen

## Licentie

AHN data is beschikbaar onder CC0 1.0 (publiek domein).
Bron: Rijkswaterstaat via PDOK.
