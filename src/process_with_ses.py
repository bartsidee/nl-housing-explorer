"""
Enhanced data processing with SES integration

⚠️ DEPRECATED: This script is no longer used in production.
Use scripts/process_multiyear_trends.py instead.

This file is kept for reference only.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_processing import DataProcessor
from scoring import ScoringEngine
from ses_loader import SESDataLoader
# from nabijheid_loader import load_nabijheid_for_dashboard  # Module removed
from data_paths import get_processed_path, MAIN_DATA_PATH


def process_data_with_ses(year_kwb: int = 2023, year_ses: int = 2022, save: bool = True) -> pd.DataFrame:
    """
    Process KWB data and merge with SES data
    
    Args:
        year_kwb: Year for KWB data
        year_ses: Year for SES data (2022 is most recent)
        save: Whether to save processed data
        
    Returns:
        DataFrame with KWB + SES data
    """
    print("=" * 60)
    print("ENHANCED DATA PROCESSING WITH SES + NABIJHEID")
    print("=" * 60)
    
    # Step 1: Process KWB data
    print(f"\n1. Processing KWB data ({year_kwb})...")
    processor = DataProcessor()
    df_kwb = processor.process_year(year=year_kwb, save=False)
    
    # Step 2: Calculate scores
    print(f"\n2. Calculating scores...")
    scorer = ScoringEngine()
    df_scored = scorer.calculate_overall_score(df_kwb)
    df_scored = scorer.add_rankings(df_scored)
    
    # Step 3: Load SES data
    print(f"\n3. Loading SES data ({year_ses})...")
    ses_loader = SESDataLoader()
    ses_df = ses_loader.load_ses_data(year=year_ses)
    
    # Step 4: Merge SES
    print(f"\n4. Merging SES data...")
    df_merged = ses_loader.merge_with_kwb_data(df_scored, ses_df)
    
    # Step 5: Load Nabijheid data (proximity indicators) - SKIPPED (module removed)
    print(f"\n5. Skipping CBS Nabijheidsstatistieken (module removed)...")
    
    # Step 7: Extract hierarchical info for bread-crumb
    print(f"\n7. Adding hierarchical navigation data...")
    df_final = add_hierarchy_info(df_merged)
    
    # Step 8: Save
    if save:
        # Save to current data location
        output_path = MAIN_DATA_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"\n✅ Saved to: {output_path}")
        print(f"   Total columns: {len(df_final.columns)}")
        print(f"   Total rows: {len(df_final)}")
    
    return df_final


def add_hierarchy_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hierarchical navigation information
    Extract gemeente/wijk names from codes for bread-crumb navigation
    """
    df = df.copy()
    
    # Extract codes based on geo_level
    # GM0014 = gemeente (6 chars)
    # WK001400 = wijk (8 chars), gemeente = GM + chars 3-6
    # BU00140000 = buurt (10 chars), gemeente = GM + chars 3-6, wijk = WK + chars 3-8
    
    def extract_gemeente_code(row):
        code = str(row['gwb_code_10'])
        if code.startswith('GM'):
            return code[:6]  # GM0014
        elif code.startswith('WK'):
            return 'GM' + code[2:6]  # WK001400 -> GM0014
        elif code.startswith('BU'):
            return 'GM' + code[2:6]  # BU00140000 -> GM0014
        return None
    
    def extract_wijk_code(row):
        code = str(row['gwb_code_10'])
        if code.startswith('WK'):
            return code[:8]  # WK001400
        elif code.startswith('BU'):
            return 'WK' + code[2:8]  # BU00140000 -> WK001400
        return None
    
    def extract_buurt_code(row):
        code = str(row['gwb_code_10'])
        if code.startswith('BU'):
            return code  # BU00140000
        return None
    
    df['gemeente_code'] = df.apply(extract_gemeente_code, axis=1)
    df['wijk_code'] = df.apply(extract_wijk_code, axis=1)
    df['buurt_code'] = df.apply(extract_buurt_code, axis=1)
    
    # Create display names (use existing regio_naam from SES or regio from KWB)
    if 'regio_naam_ses' in df.columns:
        df['display_name'] = df['regio_naam_ses'].fillna(df['regio'])
    else:
        df['display_name'] = df['regio']
    
    print(f"   Added hierarchy columns: gemeente_code, wijk_code, buurt_code")
    
    return df


def main():
    """Process data with SES integration"""
    df = process_data_with_ses(year_kwb=2023, year_ses=2022, save=True)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n📊 Final dataset:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns):,}")
    print(f"\n📋 Sample columns:")
    for i, col in enumerate(df.columns[:15], 1):
        print(f"   {i:2d}. {col}")
    if len(df.columns) > 15:
        print(f"   ... and {len(df.columns) - 15} more")
    
    # Show new nabijheid columns
    nabijheid_cols = [col for col in df.columns if col.startswith('afs_')]
    if nabijheid_cols:
        print(f"\n🚆 Nabijheid indicators ({len(nabijheid_cols)}):")
        for col in nabijheid_cols:
            completeness = (df[col].notna().sum() / len(df)) * 100
            print(f"   • {col:25s}: {completeness:5.1f}% data")


if __name__ == '__main__':
    main()
