#!/usr/bin/env python3
"""
Multi-Year KWB Data Processing for Trend Analysis

This script processes multiple years of KWB data using the existing pipeline,
then calculates real trends between years.

Strategy:
- Process complete years only (2022, 2023)
- Save each year: data/processed/{YEAR}/main_data.csv
- Calculate trends: 2022 → 2023
- Merge trends with current (2023) data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from process_with_ses import process_data_with_ses
from veiligheid_loader import load_politie_crime_data, VeiligheidLoader
from rivm_groen_loader import RIVMGroenLoader
from nabijheid_loader import NabijheidLoader
from ahn_nap_loader import AHNNAPLoader


def process_multiple_years(years: list[int], force_reprocess: bool = False, include_veiligheid: bool = True):
    """
    Process multiple years of KWB data
    
    Args:
        years: List of years to process (e.g., [2022, 2023])
        force_reprocess: If False, skip years that already exist
        include_veiligheid: If True, integrate crime data from Politie
    
    Returns:
        Dict of {year: DataFrame}
    """
    print("=" * 70)
    print("MULTI-YEAR KWB DATA PROCESSING")
    print("=" * 70)
    print(f"\nYears to process: {years}")
    print(f"Include veiligheid data: {'✅ Yes' if include_veiligheid else '❌ No'}")
    
    # Check veiligheid data availability if requested
    veiligheid_available_years = []
    if include_veiligheid:
        loader = VeiligheidLoader()
        veiligheid_available_years = loader.get_available_years()
        print(f"Veiligheid data available for years: {veiligheid_available_years}")
    
    processed_data = {}
    
    for year in years:
        print(f"\n{'=' * 70}")
        print(f"PROCESSING YEAR: {year}")
        print(f"{'=' * 70}")
        
        # Check if already processed (prefer Parquet, fallback to CSV)
        output_path_parquet = Path(f"data/processed/{year}/main_data.parquet")
        output_path_csv = Path(f"data/processed/{year}/main_data.csv")
        
        if (output_path_parquet.exists() or output_path_csv.exists()) and not force_reprocess:
            if output_path_parquet.exists():
                print(f"✓ Already processed, loading from: {output_path_parquet}")
                df = pd.read_parquet(output_path_parquet)
            else:
                print(f"✓ Already processed, loading from: {output_path_csv}")
                df = pd.read_csv(output_path_csv, low_memory=False)
            processed_data[year] = df
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Check if veiligheid data is already in the file
            has_veiligheid = 'crime_rate' in df.columns
            print(f"  Veiligheid data: {'✅ Included' if has_veiligheid else '❌ Not included'}")
            
            continue
        
        # Process this year
        try:
            print(f"\nProcessing {year} from scratch...")
            
            # Use matching SES year, fallback to 2022 if not available
            # SES data: 2014-2022 available, 2023 not yet published
            ses_year = min(year, 2022)  # Use same year or 2022 (latest available)
            
            if ses_year != year:
                print(f"   Note: Using SES data from {ses_year} (2023 not yet published)")
            
            df = process_data_with_ses(
                year_kwb=year,
                year_ses=ses_year,  # Dynamic SES year!
                save=False          # We'll save manually to year-specific folder
            )
            
            # ==========================================
            # ADD VEILIGHEID DATA
            # ==========================================
            if include_veiligheid and year in veiligheid_available_years:
                print(f"\n🚨 Loading veiligheid data for {year}...")
                
                try:
                    # Load and process crime data with population normalization
                    # KWB uses 'a_inw' for population count
                    crime_df = load_politie_crime_data(
                        year=year,
                        population_df=df[['gwb_code_10', 'a_inw']].rename(columns={'a_inw': 'inwoners'}),
                        weights=None  # Use default weights (can be customized later)
                    )
                    
                    if crime_df is not None and not crime_df.empty:
                        # Select only the crime columns to merge
                        crime_cols = [col for col in crime_df.columns 
                                     if col not in ['gwb_code_10']]
                        
                        # Merge with main data (only buurten will have data initially)
                        df = df.merge(
                            crime_df[['gwb_code_10'] + crime_cols],
                            on='gwb_code_10',
                            how='left'
                        )
                        
                        buurt_coverage = df[df['geo_level'] == 'buurt']['crime_rate'].notna().sum()
                        print(f"   ✅ Added {len(crime_cols)} veiligheid columns (buurten only)")
                        print(f"   Buurt coverage: {buurt_coverage:,} / {(df['geo_level'] == 'buurt').sum():,}")
                        
                        # ==========================================
                        # AGGREGATE VEILIGHEID TO WIJKEN & GEMEENTEN
                        # ==========================================
                        print(f"   🔄 Aggregating veiligheid to wijken & gemeenten...")
                        
                        # For wijken: aggregate from buurten
                        if 'wijk_code' in df.columns:
                            agg_dict = {
                                'totaal': 'sum',
                                'crime_rate': 'mean',
                                'a_inw': 'sum'
                            }
                            # Add inbraak if available
                            if 'inbraak' in df.columns:
                                agg_dict['inbraak'] = 'sum'
                            
                            wijk_agg = df[df['geo_level'] == 'buurt'].groupby('wijk_code').agg(agg_dict).reset_index()
                            
                            # Recalculate rates for wijken
                            wijk_agg['crime_rate_recalc'] = (wijk_agg['totaal'] / wijk_agg['a_inw'] * 1000).fillna(0)
                            if 'inbraak' in wijk_agg.columns:
                                wijk_agg['inbraak_rate_recalc'] = (wijk_agg['inbraak'] / wijk_agg['a_inw'] * 1000).fillna(0)
                            
                            # Update wijken with aggregated data
                            for _, wijk_row in wijk_agg.iterrows():
                                mask = (df['geo_level'] == 'wijk') & (df['gwb_code_10'] == wijk_row['wijk_code'])
                                if mask.any():
                                    df.loc[mask, 'totaal'] = wijk_row['totaal']
                                    df.loc[mask, 'crime_rate'] = wijk_row['crime_rate_recalc']
                                    if 'inbraak' in wijk_agg.columns:
                                        df.loc[mask, 'inbraak'] = wijk_row['inbraak']
                                        df.loc[mask, 'inbraak_rate'] = wijk_row['inbraak_rate_recalc']
                        
                        # For gemeenten: aggregate from buurten
                        if 'gemeente_code' in df.columns:
                            agg_dict = {
                                'totaal': 'sum',
                                'crime_rate': 'mean',
                                'a_inw': 'sum'
                            }
                            # Add inbraak if available
                            if 'inbraak' in df.columns:
                                agg_dict['inbraak'] = 'sum'
                            
                            gemeente_agg = df[df['geo_level'] == 'buurt'].groupby('gemeente_code').agg(agg_dict).reset_index()
                            
                            # Recalculate for gemeenten
                            gemeente_agg['crime_rate_recalc'] = (gemeente_agg['totaal'] / gemeente_agg['a_inw'] * 1000).fillna(0)
                            if 'inbraak' in gemeente_agg.columns:
                                gemeente_agg['inbraak_rate_recalc'] = (gemeente_agg['inbraak'] / gemeente_agg['a_inw'] * 1000).fillna(0)
                            
                            # Update gemeenten
                            for _, gem_row in gemeente_agg.iterrows():
                                mask = (df['geo_level'] == 'gemeente') & (df['gwb_code_10'] == gem_row['gemeente_code'])
                                if mask.any():
                                    df.loc[mask, 'totaal'] = gem_row['totaal']
                                    df.loc[mask, 'crime_rate'] = gem_row['crime_rate_recalc']
                                    if 'inbraak' in gemeente_agg.columns:
                                        df.loc[mask, 'inbraak'] = gem_row['inbraak']
                                        df.loc[mask, 'inbraak_rate'] = gem_row['inbraak_rate_recalc']
                        
                        # Handle inf values (areas with 0 population)
                        df['crime_rate'] = df['crime_rate'].replace([np.inf, -np.inf], np.nan)
                        if 'inbraak_rate' in df.columns:
                            df['inbraak_rate'] = df['inbraak_rate'].replace([np.inf, -np.inf], np.nan)
                        
                        print(f"   ✅ Veiligheid aggregated to all geo levels")
                        print(f"   Total coverage: {df['crime_rate'].notna().sum():,} / {len(df):,} "
                              f"({df['crime_rate'].notna().sum() / len(df) * 100:.1f}%)")
                    else:
                        print(f"   ⚠️  No crime data returned for {year}")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not load veiligheid data: {e}")
                    print(f"   Continuing without veiligheid data...")
            
            elif include_veiligheid and year not in veiligheid_available_years:
                print(f"\n🚨 Veiligheid data not available for {year}")
                print(f"   Download via: python3 scripts/download_veiligheid_data.py")
            
            # ==========================================
            # ADD RIVM GROEN DATA
            # ==========================================
            print(f"\n🌳 Loading RIVM groenpercentage data...")
            
            try:
                rivm_loader = RIVMGroenLoader()
                groen_df = rivm_loader.load_for_dashboard()
                
                if groen_df is not None and not groen_df.empty:
                    # Merge with main data
                    df = df.merge(
                        groen_df[['gwb_code_10', 'groen_percentage']],
                        on='gwb_code_10',
                        how='left'
                    )
                    
                    coverage = df['groen_percentage'].notna().sum()
                    print(f"   ✅ Added groenpercentage column")
                    print(f"   Coverage: {coverage:,} / {len(df):,} ({coverage/len(df)*100:.1f}%)")
                    print(f"   Range: {df['groen_percentage'].min():.1f}% - {df['groen_percentage'].max():.1f}%")
                else:
                    print(f"   ⚠️  No groen data returned")
                    
            except Exception as e:
                print(f"   ⚠️  Could not load RIVM groen data: {e}")
                print(f"   Continuing without groen data...")
            
            # ==========================================
            # ADD CBS NABIJHEID DATA (PROXIMITY)
            # ==========================================
            print(f"\n🚆 Loading CBS nabijheid (proximity) data...")
            
            try:
                nabijheid_loader = NabijheidLoader()
                # Use most recent nabijheid data (2024) for all years
                # Nabijheid data changes slowly, so we can use latest for all years
                nabijheid_df = nabijheid_loader.load_and_process(2024)
                
                if nabijheid_df is not None and not nabijheid_df.empty:
                    # Merge with main data
                    nabijheid_cols = ['gwb_code_10', 'g_afs_trein', 'g_afs_overstap', 
                                     'g_afs_oprit', 'g_afs_bieb']
                    available_cols = ['gwb_code_10'] + [c for c in nabijheid_cols[1:] 
                                                        if c in nabijheid_df.columns]
                    
                    df = df.merge(
                        nabijheid_df[available_cols],
                        on='gwb_code_10',
                        how='left'
                    )
                    
                    # Show coverage for each indicator
                    print(f"   ✅ Added nabijheid indicators:")
                    for col in available_cols[1:]:
                        if col in df.columns:
                            coverage = df[col].notna().sum()
                            mean_val = df[col].mean()
                            print(f"      {col:15s}: {coverage:>6,}/{len(df):>6,} "
                                  f"({coverage/len(df)*100:>5.1f}%) - mean: {mean_val:.2f} km")
                else:
                    print(f"   ⚠️  No nabijheid data returned")
                    
            except Exception as e:
                print(f"   ⚠️  Could not load nabijheid data: {e}")
                print(f"   Continuing without nabijheid data...")
            
            # ==========================================
            # ADD AHN NAP HOOGTE DATA
            # ==========================================
            print(f"\n🏔️  Loading AHN NAP hoogte (elevation) data...")
            
            try:
                nap_loader = AHNNAPLoader()
                # Load cached NAP data (pre-calculated via download_nap_data.py)
                nap_df = nap_loader.load_all_nap_heights(use_cache=True)
                
                if nap_df is not None and not nap_df.empty:
                    # Merge with main data
                    df = df.merge(
                        nap_df[['gwb_code_10', 'nap_hoogte_gem']],
                        on='gwb_code_10',
                        how='left'
                    )
                    
                    coverage = df['nap_hoogte_gem'].notna().sum()
                    print(f"   ✅ Added nap_hoogte_gem column")
                    print(f"   Coverage: {coverage:,} / {len(df):,} ({coverage/len(df)*100:.1f}%)")
                    if coverage > 0:
                        print(f"   Range: {df['nap_hoogte_gem'].min():.2f}m - {df['nap_hoogte_gem'].max():.2f}m NAP")
                        print(f"   Mean: {df['nap_hoogte_gem'].mean():.2f}m NAP")
                else:
                    print(f"   ⚠️  No NAP data returned")
                    
            except Exception as e:
                print(f"   ⚠️  Could not load NAP data: {e}")
                print(f"   Continuing without NAP data...")
                print(f"   To download NAP data, run: python scripts/download_nap_data.py")
            
            # ==========================================
            # BACKFILL MISSING DATA FROM PREVIOUS YEAR
            # ==========================================
            if year >= 2024:
                print(f"\n🔄 Checking for missing data in {year}...")
                
                # Indicators that may be missing in newer years
                backfill_candidates = [
                    'g_ink_pi',      # Gemiddeld inkomen
                    'p_arb_pp',      # Participatiegraad
                    'p_ink_li',      # Laag inkomen
                    'p_ink_hi',      # Hoog inkomen
                    # Add distance indicators (often missing for new buurten)
                    'g_afs_sc',      # Afstand basisschool
                    'g_afs_kv',      # Afstand kinderopvang
                    'g_afs_hp',      # Afstand huisarts
                    'g_afs_gs',      # Afstand supermarkt
                ]
                
                # Check which columns have 0% or very low coverage
                missing_cols = []
                partial_cols = []  # Columns with some but not full coverage
                
                for col in backfill_candidates:
                    if col in df.columns:
                        coverage = df[col].notna().sum()
                        coverage_pct = coverage / len(df) * 100
                        missing_count = df[col].isna().sum()
                        
                        if coverage_pct < 5:  # Less than 5% coverage
                            missing_cols.append(col)
                        elif missing_count > 0 and coverage_pct < 99:  # Some missing values
                            partial_cols.append((col, missing_count))
                
                # Strategy 1: Backfill completely missing columns
                if missing_cols:
                    print(f"   Found {len(missing_cols)} columns with <5% coverage:")
                    for col in missing_cols:
                        print(f"      - {col}")
                    
                    # Try to load 2023 data for backfilling
                    backfill_year = 2023
                    backfill_path_parquet = Path("data/processed") / str(backfill_year) / 'main_data.parquet'
                    backfill_path_csv = Path("data/processed") / str(backfill_year) / 'main_data.csv'
                    
                    if backfill_path_parquet.exists():
                        print(f"   📥 Loading {backfill_year} data for backfilling...")
                        df_backfill = pd.read_parquet(backfill_path_parquet)
                    elif backfill_path_csv.exists():
                        print(f"   📥 Loading {backfill_year} data for backfilling (CSV)...")
                        df_backfill = pd.read_csv(backfill_path_csv, low_memory=False)
                        
                        # Select only the missing columns + gwb_code_10
                        backfill_cols_available = ['gwb_code_10'] + [c for c in missing_cols 
                                                                      if c in df_backfill.columns]
                        
                        if len(backfill_cols_available) > 1:  # More than just gwb_code_10
                            # Remove existing (empty) columns
                            df = df.drop(columns=[c for c in missing_cols if c in df.columns])
                            
                            # Merge backfill data
                            df = df.merge(
                                df_backfill[backfill_cols_available],
                                on='gwb_code_10',
                                how='left'
                            )
                            
                            print(f"   ✅ Backfilled {len(backfill_cols_available)-1} columns from {backfill_year}:")
                            for col in backfill_cols_available[1:]:
                                if col in df.columns:
                                    coverage = df[col].notna().sum()
                                    coverage_pct = coverage / len(df) * 100
                                    print(f"      ✅ {col:15s}: {coverage:>6,}/{len(df):>6,} "
                                          f"({coverage_pct:>5.1f}%) [from {backfill_year}]")
                        else:
                            print(f"   ⚠️  No backfill data available in {backfill_year}")
                    else:
                        print(f"   ⚠️  {backfill_year} data not found")
                
                # Strategy 2: Fill partial gaps (e.g., new buurten without distance data)
                if partial_cols:
                    print(f"\n   Found {len(partial_cols)} columns with partial coverage:")
                    for col, missing_count in partial_cols:
                        print(f"      - {col}: {missing_count} missing values")
                    
                    backfill_year = 2024  # Try most recent year first
                    backfill_path_parquet = Path("data/processed") / str(backfill_year) / 'main_data.parquet'
                    backfill_path_csv = Path("data/processed") / str(backfill_year) / 'main_data.csv'
                    
                    if backfill_path_parquet.exists() or backfill_path_csv.exists():
                        print(f"   📥 Loading {backfill_year} data for gap-filling...")
                        if backfill_path_parquet.exists():
                            df_backfill = pd.read_parquet(backfill_path_parquet)
                        else:
                            df_backfill = pd.read_csv(backfill_path_csv, low_memory=False)
                        
                        filled_any = False
                        for col, missing_count in partial_cols:
                            if col in df_backfill.columns:
                                # Only fill NaN values, keep existing values
                                mask = df[col].isna()
                                if mask.sum() > 0:
                                    # Merge and fill only NaN positions
                                    df.loc[mask, col] = df.loc[mask, 'gwb_code_10'].map(
                                        df_backfill.set_index('gwb_code_10')[col]
                                    )
                                    
                                    filled_count = mask.sum() - df[col].isna().sum()
                                    if filled_count > 0:
                                        filled_any = True
                                        coverage = df[col].notna().sum()
                                        coverage_pct = coverage / len(df) * 100
                                        print(f"      ✅ {col:15s}: filled {filled_count} gaps → "
                                              f"{coverage:>6,}/{len(df):>6,} ({coverage_pct:>5.1f}%)")
                        
                        if not filled_any:
                            print(f"      ℹ️  No additional data found in {backfill_year}")
                    else:
                        print(f"      ⚠️  {backfill_year} data not found")
                
                if not missing_cols and not partial_cols:
                    print(f"   ✅ No missing data detected - all indicators have good coverage")
            
            # Save to year-specific folder (Parquet only)
            output_path_parquet = Path(f"data/processed/{year}/main_data.parquet")
            output_path_parquet.parent.mkdir(parents=True, exist_ok=True)
            
            # Save Parquet
            df.to_parquet(output_path_parquet, compression='gzip', index=False)
            
            parquet_size_mb = output_path_parquet.stat().st_size / (1024 * 1024)
            
            print(f"\n✅ Saved {year} data: {output_path_parquet} ({parquet_size_mb:.1f} MB)")
            print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")
            
            processed_data[year] = df
            
        except Exception as e:
            print(f"\n❌ Error processing {year}: {e}")
            print(f"   Skipping this year...")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 70}")
    print(f"✅ Processed {len(processed_data)} years successfully")
    print(f"{'=' * 70}\n")
    
    return processed_data


def calculate_year_over_year_trends(
    df_old: pd.DataFrame, 
    df_new: pd.DataFrame,
    year_old: int,
    year_new: int
) -> pd.DataFrame:
    """
    Calculate trends between two consecutive years
    
    Returns DataFrame with trend columns
    """
    print(f"\n📈 Calculating trends: {year_old} → {year_new}")
    
    # Indicators to track (must exist in both years)
    indicators = [
        'pct_children',
        'pct_families',
        'bev_dich',
        'ses_overall',
        'ses_onderwijs',
        'g_ink_pi',
        'p_arb_pp',
        'groen_percentage',
        'p_koopw',
        'g_hhgro',
        'p_ink_li',
        'area_per_person_m2',
        'crime_rate',  # Veiligheid: totaal misdrijven per 1000 inwoners
        'inbraak_rate',  # Veiligheid: inbraak per 1000 inwoners
        'p_herk_nl',  # Herkomst: % Nederlandse herkomst
        'p_herk_eur',  # Herkomst: % Europese herkomst
        'p_herk_neu',  # Herkomst: % Buiten-Europese herkomst
    ]
    
    # Only use indicators present in BOTH dataframes
    available_indicators = [ind for ind in indicators 
                           if ind in df_old.columns and ind in df_new.columns]
    
    print(f"   Tracking {len(available_indicators)} indicators")
    
    trends = pd.DataFrame()
    trends['gwb_code_10'] = df_new['gwb_code_10']
    
    for indicator in available_indicators:
        # Merge old and new values
        merged = df_old[['gwb_code_10', indicator]].merge(
            df_new[['gwb_code_10', indicator]],
            on='gwb_code_10',
            suffixes=('_old', '_new'),
            how='inner'
        )
        
        # Calculate absolute change
        merged[f'trend_{indicator}_abs'] = (
            merged[f'{indicator}_new'] - merged[f'{indicator}_old']
        )
        
        # Calculate percentage change
        merged[f'trend_{indicator}_pct'] = np.where(
            merged[f'{indicator}_old'] != 0,
            (merged[f'trend_{indicator}_abs'] / merged[f'{indicator}_old']) * 100,
            np.nan
        )
        
        # Determine direction based on years span
        years_diff = year_new - year_old
        
        # Adjust threshold based on time period
        # For 1 year: 2% is significant
        # For 2+ years: scale proportionally (2% per year = 4% for 2 years)
        threshold = 2.0 * years_diff  # 2% per year
        
        merged[f'trend_{indicator}_dir'] = 'stabiel'
        merged.loc[merged[f'trend_{indicator}_pct'] > threshold, f'trend_{indicator}_dir'] = 'stijgend'
        merged.loc[merged[f'trend_{indicator}_pct'] < -threshold, f'trend_{indicator}_dir'] = 'dalend'
        
        # Add to trends dataframe
        trends = trends.merge(
            merged[['gwb_code_10', f'trend_{indicator}_abs', 
                   f'trend_{indicator}_pct', f'trend_{indicator}_dir']],
            on='gwb_code_10',
            how='left'
        )
        
        # Summary stats
        stijgend = (merged[f'trend_{indicator}_dir'] == 'stijgend').sum()
        dalend = (merged[f'trend_{indicator}_dir'] == 'dalend').sum()
        stabiel = (merged[f'trend_{indicator}_dir'] == 'stabiel').sum()
        avg_pct = merged[f'trend_{indicator}_pct'].mean()
        
        print(f"   • {indicator:25s}: ↑{stijgend:5d} ↓{dalend:5d} ↔{stabiel:5d}  (gem: {avg_pct:+6.2f}%)")
    
    return trends


def calculate_composite_trend_score(trends_df: pd.DataFrame, custom_weights: dict = None) -> pd.Series:
    """
    Calculate weighted composite trend score
    
    Args:
        trends_df: DataFrame with trend columns
        custom_weights: Optional dict with custom weights
                       If None, use default family/growth focus
    
    Returns:
        Series with composite trend scores
    """
    print("\n📊 Calculating composite trend score...")
    
    # Default weights (family/growth focus) if no custom weights provided
    if custom_weights is None:
        weights = {
            'pct_children': 3.0,
            'pct_families': 2.0,
            'bev_dich': 1.5,
            'ses_overall': 1.0,
            'p_arb_pp': 1.0,
            'groen_percentage': 0.5,
        }
        print("   Using DEFAULT weights (family/growth focus)")
    else:
        # Use custom weights
        weights = custom_weights.copy()
        print("   Using CUSTOM weights")
    
    score = pd.Series(0.0, index=trends_df.index)
    total_weight = 0
    indicators_used = []
    
    for indicator, weight in weights.items():
        col = f'trend_{indicator}_pct'
        if col in trends_df.columns and weight > 0:
            # Normalize and clip extreme values
            normalized = trends_df[col].fillna(0).clip(-50, 50)
            score += normalized * weight
            total_weight += weight
            indicators_used.append(indicator)
    
    if total_weight > 0:
        score = score / total_weight
    else:
        score = pd.Series(np.nan, index=trends_df.index)
    
    # Summary
    if score.notna().sum() > 0:
        print(f"   ✓ Calculated for {score.notna().sum():,} locations")
        print(f"   ✓ Using {len(indicators_used)} indicators: {', '.join(indicators_used)}")
        print(f"   Range: {score.min():.1f} to {score.max():.1f}")
        print(f"   Mean: {score.mean():.1f}")
        
        # Distribution
        stijgend = (score > 5).sum()
        stabiel = ((score >= -5) & (score <= 5)).sum()
        dalend = (score < -5).sum()
        print(f"   Distribution: ↑{stijgend:,} ↔{stabiel:,} ↓{dalend:,}")
    
    return score


def main():
    """
    Main workflow:
    1. Process multiple years of KWB data
    2. Calculate trends (longer period = better signal)
    3. Merge with current data
    4. Save
    """
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process multi-year KWB data with trends',
        epilog='Use --force to regenerate all year files. Use --years to process specific years only.'
    )
    parser.add_argument('--force', '-f', action='store_true', help='Force reprocess all years')
    parser.add_argument('--years', '-y', nargs='+', type=int, default=[2020, 2021, 2022, 2023, 2024, 2025], help='Years to process')
    parser.add_argument('--no-veiligheid', action='store_true', help='Skip crime data')
    args = parser.parse_args()
    
    years_to_process = sorted(args.years)
    force_reprocess = args.force
    include_veiligheid = not args.no_veiligheid
    
    print("\n📋 STRATEGY:")
    print(f"   1. Process years: {years_to_process}")
    print(f"   2. Calculate trends: {years_to_process[0]} → {years_to_process[-1]} ({len(years_to_process)-1} years)")
    print(f"   3. Use {years_to_process[-1]} as current/baseline")
    print(f"   4. Merge trends into current data")
    print(f"   5. Threshold: 2% per year (so {2 * (len(years_to_process)-1)}% for full period)\n")
    
    # Step 1: Process all years
    processed_data = process_multiple_years(years_to_process, force_reprocess=force_reprocess, include_veiligheid=include_veiligheid)
    
    if len(processed_data) < 2:
        print("\n❌ Need at least 2 years of data to calculate trends!")
        return 1
    
    # Step 2: Calculate trends
    # Use FIRST and LAST year for longest trend period
    years = sorted(processed_data.keys())
    df_old = processed_data[years[0]]   # First year (2020)
    df_new = processed_data[years[-1]]  # Last year (2023)
    
    print(f"\n📊 Using {len(years)} years of data ({years[0]}-{years[-1]})")
    print(f"   Trend calculation: {years[0]} → {years[-1]} ({years[-1] - years[0]} years)")
    
    trends = calculate_year_over_year_trends(
        df_old, df_new,
        year_old=years[0],
        year_new=years[-1]
    )
    
    # Step 3: Calculate composite trend score
    print("\n5. Calculating composite trend score...")
    
    # Use default weights (custom weights are managed in the web UI via session state)
    trends['trend_score'] = calculate_composite_trend_score(trends, custom_weights=None)
    
    # Step 4: Merge trends with current data
    print(f"\n🔗 Merging trends with current ({years[-1]}) data...")
    df_with_trends = df_new.merge(trends, on='gwb_code_10', how='left')
    
    trend_cols = [col for col in trends.columns if col.startswith('trend_')]
    print(f"   Added {len(trend_cols)} trend columns")
    
    # Step 5: Save
    print("\n💾 Saving results...")
    
    # Main output: current with trends (Parquet only)
    output_path_parquet = Path("data/processed/current/main_data_with_trends.parquet")
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    # Save Parquet
    df_with_trends.to_parquet(output_path_parquet, compression='gzip', index=False)
    parquet_size_mb = output_path_parquet.stat().st_size / (1024 * 1024)
    print(f"   ✅ {output_path_parquet} ({parquet_size_mb:.1f} MB)")
    
    # Metadata
    import json
    from datetime import datetime
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "base_year": years[-1],
        "trend_years": f"{years[0]}-{years[-1]}",
        "trend_period_years": years[-1] - years[0],
        "rows": len(df_with_trends),
        "columns": len(df_with_trends.columns),
        "trend_columns": len(trend_cols),
        "note": f"Real trends calculated from actual multi-year CBS KWB data ({years[0]}-{years[-1]})",
        "threshold_pct_per_year": 2.0
    }
    
    metadata_path = output_path_parquet.parent / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ {metadata_path}")
    
    print("\n" + "=" * 70)
    print("✅ REAL TREND ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Base year: {years[-1]}")
    print(f"   Trend period: {years[0]} → {years[-1]} ({years[-1] - years[0]} years)")
    print(f"   Locations: {len(df_with_trends):,}")
    print(f"   Indicators tracked: {len([c for c in trend_cols if c.endswith('_pct')])}")
    print(f"   Trend scores: {df_with_trends['trend_score'].notna().sum():,}")
    print(f"   Threshold: 2% per year × {years[-1] - years[0]} = {2 * (years[-1] - years[0])}% total")
    print(f"\n💾 Output: {output_path_parquet} ({parquet_size_mb:.1f} MB)")
    print(f"\n🚀 Dashboard is ready to use REAL {years[-1] - years[0]}-year trend data!")
    print(f"   Run: streamlit run app.py\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
