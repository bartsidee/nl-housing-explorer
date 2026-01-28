# Geo Cache Management

Deze folder bevat gecachte geografische data om herhaalde API calls te voorkomen.

## Folder Structuur

```
cache/
├── README.md                           # Deze file
├── gemeenten_2024.geojson             # ~342 gemeenten 
├── wijken_2024_all.geojson            # ~3,300 wijken 
├── buurten_2024_all.geojson           # ~14,500 buurten 
└── ahn/                               # NAP hoogte cache
    ├── gemeente_nap_heights.csv       
    ├── wijk_nap_heights_all.csv       
    └── buurt_nap_heights_all.csv      
```

## Cache Schaalbaarheid

### Huidige Situatie (Optimaal)

De cache is goed geoptimaliseerd voor schaalbaarheid:

#### ✅ GeoJSON Cache (Geometrieën)
- **Gemeenten**: 15 MB voor 342 records - prima
- **Wijken**: 150 MB voor 3,300 records - acceptabel
- **Buurten**: 500 MB voor 14,500 records - **groot maar werkbaar**

**Waarom dit werkt:**
1. Eenmalig download, daarna snel van disk laden
2. GeoJSON wordt alleen gebruikt voor kaartweergave
3. Geometrieën worden geladen met `gpd.read_file()` wat efficiënt is
4. Per gemeente caching (bijv. `buurten_2024_GM0363.geojson`) is veel kleiner

#### ✅ NAP Hoogte Cache (Tabular Data)
- **Gemeenten**: 100 KB - perfect
- **Wijken**: 300 KB - perfect  
- **Buurten**: 1 MB - perfect

**Waarom dit optimaal is:**
1. CSV is veel compacter dan GeoJSON (geen geometrie opslag)
2. Alleen 3 kolommen: code, naam, hoogte
3. Snel laden met `pd.read_csv()`
4. Batch processing met checkpoints voor grote datasets

### Potentiële Problemen & Oplossingen

#### ⚠️ Probleem 1: Grote GeoJSON Bestanden

**Symptoom**: `buurten_2024_all.geojson` is 500 MB

**Oplossingen (indien nodig in toekomst):**

1. **Simplified geometries** (trade-off: minder detail)
   ```python
   gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)
   ```

2. **GeoParquet format** (moderne, compacte alternatief)
   ```python
   # In plaats van GeoJSON
   gdf.to_parquet('buurten_2024_all.parquet')
   gdf = gpd.read_parquet('buurten_2024_all.parquet')
   # ~70% grootte reductie
   ```

3. **Spatial database** (voor zeer grote datasets)
   ```python
   # PostGIS of GeoPackage
   gdf.to_file('cache.gpkg', layer='buurten', driver='GPKG')
   ```

4. **Per provincie caching** (splits 14,500 buurten op)
   - 12 bestanden van ~40 MB i.p.v. 1 van 500 MB
   - Alleen laden wat nodig is

#### ⚠️ Probleem 2: Memory Usage

**Symptoom**: App gebruikt veel geheugen bij laden van alle buurten

**Huidige mitigaties:**
- Streamlit `@st.cache_data` voorkomt herhaald laden
- Lazy loading: alleen laden wat getoond wordt
- Geometrie simplificatie in kaartweergave

**Extra oplossingen indien nodig:**
```python
# 1. Streaming read (chunks)
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    filtered = chunk[chunk['column'] > threshold]
    chunks.append(filtered)

# 2. Selective loading
gdf = gpd.read_file('buurten.geojson', 
                    bbox=(minx, miny, maxx, maxy))  # Spatial filter

# 3. Column selection
df = pd.read_csv('data.csv', usecols=['code', 'naam', 'value'])
```

### Batch Processing voor NAP Data

Voor grote datasets (>1000 buurten) gebruikt de NAP loader **batch processing**:

```python
# Automatisch bij > 1000 records
loader.load_or_calculate_buurt_nap(
    batch_size=100  # Process 100 buurten per batch
)
```

**Features:**
- ✅ Checkpoints: hervatbaar na onderbreking
- ✅ Progress tracking per batch
- ✅ Memory efficient: proces in kleine chunks
- ✅ Geen data verlies bij crash

**Checkpoint bestanden:**
- `buurt_nap_heights_all_checkpoint.csv` - tijdelijk
- Automatisch verwijderd na succesvolle completion
- Hervatten: run script opnieuw, het detecteert checkpoint

## Cache Invalidatie

### Wanneer cache vernieuwen?

1. **Gemeenten/Wijken/Buurten**: Jaarlijks (CBS wijzigt grenzen 1x per jaar)
2. **NAP hoogtes**: Eenmalig (verandert niet)

### Handmatig vernieuwen

```bash
# Verwijder specifieke cache
rm data/geo/cache/gemeenten_2024.geojson

# Verwijder alle geo cache
rm data/geo/cache/*.geojson

# Verwijder NAP cache
rm data/geo/cache/ahn/*.csv

# Force redownload
python scripts/download_nap_data.py --force --level all
```

### Automatisch vernieuwen

In code:
```python
# Force fresh download
geo_loader = PDOKGeoLoader()
gdf = geo_loader.get_gemeenten(use_cache=False)

# NAP
nap_loader = AHNNAPLoader()
df = nap_loader.load_gemeente_nap(force_recalculate=True)
```

## Performance Benchmarks

### Typische Load Tijden

| Bestand | Grootte | Load tijd (SSD) | Load tijd (HDD) |
|---------|---------|-----------------|-----------------|
| gemeenten GeoJSON | 15 MB | 0.5s | 2s |
| wijken GeoJSON | 150 MB | 3s | 15s |
| buurten GeoJSON | 500 MB | 10s | 45s |
| gemeente NAP CSV | 100 KB | <0.1s | <0.1s |
| wijk NAP CSV | 300 KB | <0.1s | <0.1s |
| buurt NAP CSV | 1 MB | 0.2s | 0.5s |

### Download Tijden (initieel)

| Dataset | Records | Download tijd |
|---------|---------|---------------|
| Gemeenten geo | 342 | ~30s |
| Wijken geo | 3,300 | ~2 min |
| Buurten geo | 14,500 | ~5 min |
| Gemeenten NAP | 342 | ~5 min |
| Wijken NAP | 3,300 | ~15 min |
| Buurten NAP | 14,500 | ~45 min |

**Totaal eerste download: ~70 minuten**
**Hergebruik: seconden tot minuten**

## Best Practices

### Voor Ontwikkeling
```bash
# Start met klein sample
python scripts/download_nap_data.py --test

# Download alleen gemeenten
python scripts/download_nap_data.py --level gemeente

# Test app met beperkte data eerst
```

### Voor Productie
```bash
# Download alles vooraf (één keer)
python scripts/download_nap_data.py --level all

# Vertrouw op cache voor runtime performance
# Gebruik --force alleen bij CBS data updates
```

### Voor CI/CD
```bash
# Commit NAP cache files (klein: 1-2 MB)
git add data/geo/cache/ahn/*.csv

# Voeg GeoJSON toe aan .gitignore (te groot voor git)
echo "data/geo/cache/*.geojson" >> .gitignore

# Download GeoJSON in deployment script
python -c "from src.pdok_geo_loader import PDOKGeoLoader; PDOKGeoLoader().get_gemeenten()"
```

## Monitoring Cache Health

### Check cache status
```bash
# List cache files with sizes
ls -lh data/geo/cache/
ls -lh data/geo/cache/ahn/

# Check if cache is complete
test -f data/geo/cache/gemeenten_2024.geojson && echo "✅ Gemeenten cached"
test -f data/geo/cache/ahn/gemeente_nap_heights.csv && echo "✅ NAP gemeenten cached"
```

### Python check
```python
from pathlib import Path

cache_dir = Path('data/geo/cache')
ahn_cache = cache_dir / 'ahn'

# Check GeoJSON cache
geo_files = ['gemeenten_2024.geojson', 
             'wijken_2024_all.geojson', 
             'buurten_2024_all.geojson']

for file in geo_files:
    if (cache_dir / file).exists():
        size_mb = (cache_dir / file).stat().st_size / 1024 / 1024
        print(f"✅ {file}: {size_mb:.1f} MB")
    else:
        print(f"❌ {file}: Missing")

# Check NAP cache
nap_files = ['gemeente_nap_heights.csv',
             'wijk_nap_heights_all.csv',
             'buurt_nap_heights_all.csv']

for file in nap_files:
    if (ahn_cache / file).exists():
        size_kb = (ahn_cache / file).stat().st_size / 1024
        print(f"✅ {file}: {size_kb:.1f} KB")
    else:
        print(f"❌ {file}: Missing")
```

## Conclusie

De huidige cache strategie is **goed geschaald** voor Nederlandse geo data:

✅ GeoJSON cache: werkbaar voor 14,500 buurten (500 MB eenmalig)
✅ NAP CSV cache: optimaal klein (1 MB totaal)
✅ Batch processing: checkpoint systeem voorkomt data verlies
✅ Lazy loading: alleen laden wat nodig is

**Geen actie nodig** tenzij:
- Disk space < 1 GB beschikbaar → overweeg GeoParquet
- Memory issues → implementeer spatial filtering
- Deployment op cloud → pre-download cache in build step
