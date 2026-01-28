#!/usr/bin/env python3
"""
Download crime data from Politie Open Data portal

This script downloads crime statistics (misdrijven) per wijk/buurt
from data.politie.nl for multiple years.

Usage:
    python3 scripts/download_veiligheid_data.py
"""

import pandas as pd
import requests
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_politie_data_manual_instructions():
    """
    Provide manual download instructions since the API requires
    interactive exploration
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║  POLITIE MISDRIJFDATA DOWNLOADEN                            ║
╚══════════════════════════════════════════════════════════════╝

De Politie data moet handmatig gedownload worden via de portal.

STAPPEN:

1. Ga naar: https://data.politie.nl

2. Klik op "Geregistreerde misdrijven" → "Per wijk en buurt"
   
   Direct link:
   https://data.politie.nl/#/Politie/nl/dataset/47022NED/table

3. FILTER INSTELLINGEN:
   - Periode: Selecteer jaren 2020, 2021, 2022, 2023, 2024, 2025
   - Gebied: "Alle wijken en buurten"
   - Misdrijf soort: "Alle categorieën" OF selecteer:
     * Totaal geregistreerde misdrijven
     * Diefstal/inbraak woning
     * Geweldsmisdrijven
     * Diefstal/diefstal uit/vanaf auto
     * Straatroof
     * Diefstal overig
     * Vernieling en beschadiging
     
4. DOWNLOAD:
   - Klik op "Download" knop (rechts boven)
   - Kies format: "CSV" (met puntkomma)
   - Sla op als: politie_misdrijven_JAAR.csv

5. BEWAAR BESTANDEN:
   data/raw/veiligheid/2020/politie_misdrijven_2020.csv
   data/raw/veiligheid/2021/politie_misdrijven_2021.csv
   data/raw/veiligheid/2022/politie_misdrijven_2022.csv
   data/raw/veiligheid/2023/politie_misdrijven_2023.csv
   data/raw/veiligheid/2024/politie_misdrijven_2024.csv
   data/raw/veiligheid/2025/politie_misdrijven_2025.csv

6. RUN PROCESSING:
   python3 scripts/process_multiyear_trends.py

╔══════════════════════════════════════════════════════════════╗
║  ALTERNATIVE: API DOWNLOAD (ADVANCED)                        ║
╚══════════════════════════════════════════════════════════════╝

De data.politie.nl portal gebruikt OData API:
https://data.politie.nl/odata/47022NED/...

Voor automatische downloads, zie documentatie:
https://data.politie.nl/portal.html?_catalog=Politie&_la=nl

""")


def try_api_download(year: int = 2023, save_dir: str = 'data/raw/veiligheid'):
    """
    Download via CBS OData Feed API
    
    API: https://dataderden.cbs.nl/ODataFeed/odata/47022NED
    Table: 47022NED (Geregistreerde misdrijven per wijk/buurt)
    
    Strategy:
    - Download 12 months separately (each month < 10k limit)
    - Aggregate to yearly totals
    - Decode CBS codes to readable names
    """
    print(f"\n🔄 API download voor jaar {year}...")
    
    # CBS OData Feed API (10k limit per request, monthly pagination works!)
    base_url = "https://dataderden.cbs.nl/ODataFeed/odata/47022NED"
    
    try:
        # Test API connectivity
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ CBS OData Feed API bereikbaar!")
        else:
            print(f"❌ API response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API niet bereikbaar: {e}")
        return False
    
    print(f"📥 Downloading {year}... (12 maanden, ~2 min)")
    
    try:
        # Download all 12 months
        data_url = f"{base_url}/TypedDataSet"
        all_monthly_data = []
        
        for month in range(1, 13):
            month_code = f"{year}MM{month:02d}"
            
            # Pagination: fetch all data for this month
            skip = 0
            page_size = 10000
            month_data = []
            
            while True:
                # Download 2 crime types: Totaal (0.0.0) + Inbraak woning (1.1.1)
                params = {
                    '$filter': f"Perioden eq '{month_code}' and (SoortMisdrijf eq '0.0.0 ' or SoortMisdrijf eq '1.1.1 ')",
                    '$format': 'json',
                    '$top': page_size,
                    '$skip': skip
                }
                
                print(f"   Maand {month:2d}/12 ({month_code}) - offset {skip}...", end='\r')
                
                try:
                    response = requests.get(data_url, params=params, timeout=60)
                    
                    if response.status_code != 200:
                        print(f"\n   ⚠️  Maand {month} failed: {response.status_code}")
                        break
                    
                    data = response.json()
                    
                    if 'value' in data and len(data['value']) > 0:
                        month_data.extend(data['value'])
                        
                        # If we got less than page_size, we're done with this month
                        if len(data['value']) < page_size:
                            break
                        
                        skip += page_size
                    else:
                        break
                        
                except Exception as e:
                    print(f"\n   ⚠️  Maand {month} error: {e}")
                    break
            
            print(f"   Maand {month:2d}/12 ({month_code}): {len(month_data):,} records")
            all_monthly_data.extend(month_data)
        
        print(f"\n✅ Downloaded {len(all_monthly_data):,} maandrecords")
        
        if len(all_monthly_data) == 0:
            print(f"⚠️  Geen data voor jaar {year}")
            return False
        
        # Convert to DataFrame
        df_monthly = pd.DataFrame(all_monthly_data)
        print(f"   Kolommen: {list(df_monthly.columns)[:5]}...")
        
        # Aggregate monthly to yearly totals
        print(f"\n📊 Aggregeren naar jaarcijfers...")
        
        # Find value column
        value_col = [col for col in df_monthly.columns 
                    if 'Geregistreerde' in col or 'GeregistreerdeMisdrijven' in col]
        if not value_col:
            print(f"   ⚠️  Geen value kolom gevonden: {df_monthly.columns.tolist()}")
            return False
        
        value_col = value_col[0]
        df_monthly[value_col] = pd.to_numeric(df_monthly[value_col], errors='coerce').fillna(0)
        
        # Group by area + crime type, sum across months
        df_yearly = df_monthly.groupby(
            ['WijkenEnBuurten', 'SoortMisdrijf'], 
            as_index=False
        )[value_col].sum()
        
        print(f"   {len(df_yearly):,} unieke combinaties")
        
        # Keep CBS codes (compatible with KWB data)
        print(f"\n✅ Behoud CBS codes (compatible met KWB gwb_code_10)")
        
        # Rename columns and add year
        df_yearly['Perioden'] = year
        df_yearly = df_yearly.rename(columns={value_col: 'GeregistreerdeMisdrijven'})
        
        # Final clean - keep codes
        df_final = df_yearly[['WijkenEnBuurten', 'SoortMisdrijf', 'Perioden', 'GeregistreerdeMisdrijven']]
        df_final = df_final[df_final['WijkenEnBuurten'].notna()]
        
        # Show unique values
        print(f"   Unieke gebiedcodes: {df_final['WijkenEnBuurten'].nunique():,}")
        print(f"   Unieke misdrijftypes: {df_final['SoortMisdrijf'].nunique()}")
        
        # Save
        save_path = Path(save_dir) / str(year) / f'politie_misdrijven_{year}.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(save_path, sep=';', index=False, encoding='utf-8')
        
        print(f"\n✅ Saved: {save_path}")
        print(f"   Size: {save_path.stat().st_size / 1024:.0f} KB")
        print(f"   Buurten: {df_final['WijkenEnBuurten'].nunique():,}")
        print(f"   Crime types: {df_final['SoortMisdrijf'].nunique()}")
        
        # Show sample
        print(f"\n📊 Sample (eerste 3 rijen):")
        sample = df_final.head(3)
        for _, row in sample.iterrows():
            print(f"   {row['WijkenEnBuurten'][:30]:30s} | {row['SoortMisdrijf'][:35]:35s} | {row['GeregistreerdeMisdrijven']:5.0f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_downloaded_files():
    """Check which years have been downloaded"""
    veiligheid_dir = Path('data/raw/veiligheid')
    
    print("\n📁 Controleer gedownloade bestanden:\n")
    
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    found = []
    
    for year in years:
        year_dir = veiligheid_dir / str(year)
        possible_files = [
            year_dir / f'politie_misdrijven_{year}.csv',
            year_dir / 'politie_misdrijven.csv',
            year_dir / f'47022NED_{year}.csv',
        ]
        
        file_found = None
        for f in possible_files:
            if f.exists():
                file_found = f
                break
        
        if file_found:
            print(f"   ✅ {year}: {file_found.name} ({file_found.stat().st_size / 1024:.1f} KB)")
            found.append(year)
        else:
            print(f"   ❌ {year}: Niet gevonden")
    
    print(f"\n📊 Status: {len(found)}/{len(years)} jaren beschikbaar")
    
    if len(found) == len(years):
        print("\n✅ Alle data beschikbaar! Je kunt nu de processing runnen:")
        print("   python3 scripts/process_multiyear_trends.py")
    else:
        print(f"\n⚠️  Nog {len(years) - len(found)} jaar(jaren) nodig voor volledige trend analyse")
    
    return found


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Politie misdrijfdata')
    parser.add_argument('--years', nargs='+', type=int, default=[2020, 2021, 2022, 2023, 2024, 2025],
                       help='Years to download (default: 2020-2025)')
    parser.add_argument('--api', action='store_true', 
                       help='Try automatic API download (experimental)')
    parser.add_argument('--manual', action='store_true',
                       help='Show manual download instructions only')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🚨 VEILIGHEIDDATA DOWNLOADER                               ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Check existing files
    existing = check_downloaded_files()
    
    if args.manual:
        # Only show manual instructions
        download_politie_data_manual_instructions()
    elif args.api:
        # Try API download for all requested years
        print("\n" + "="*60)
        print("AUTOMATISCHE API DOWNLOAD")
        print("="*60)
        
        success_count = 0
        for year in args.years:
            if year in existing:
                print(f"\n⏭️  {year}: Already downloaded, skipping")
                success_count += 1
                continue
            
            if try_api_download(year):
                print(f"✅ {year}: Download succesvol!")
                success_count += 1
            else:
                print(f"❌ {year}: Download failed")
        
        print(f"\n📊 Summary: {success_count}/{len(args.years)} years downloaded")
        
        if success_count < len(args.years):
            print("\n⚠️  Some downloads failed. Try manual method:")
            print("   python3 scripts/download_veiligheid_data.py --manual")
    else:
        # Default: show instructions
        download_politie_data_manual_instructions()
        
        # Also try API for one year as demo
        print("\n" + "="*60)
        print("API DOWNLOAD DEMO (2023)")
        print("="*60)
        print("💡 Tip: Use --api flag to download all years automatically")
        print("   Example: python3 scripts/download_veiligheid_data.py --api")
        
        if 2023 not in existing:
            if try_api_download(2023):
                print("\n✅ API download werkt! Je kunt nu alle jaren downloaden:")
                print("   python3 scripts/download_veiligheid_data.py --api --years 2020 2021 2022 2023")
            else:
                print("\n⚠️  API download mislukt, gebruik handmatige methode")
    
    print("\n✅ Script completed")
