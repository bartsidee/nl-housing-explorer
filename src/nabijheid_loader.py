"""
Nabijheid Data Loader

Loads and processes proximity (nabijheid) statistics from CBS for integration
with KWB data. Includes transport and amenity proximity indicators.

Data source: CBS Nabijheidsstatistieken
Author: AI Assistant
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

import warnings
warnings.filterwarnings('ignore')


class NabijheidLoader:
    """Load and process CBS proximity (nabijheid) data"""
    
    # Transport and amenity indicators we want to load
    INDICATORS = {
        'g_afs_trein': 'D000052',      # Afstand tot treinstations totaal
        'g_afs_overstap': 'D000014',   # Afstand tot belangrijk overstapstation
        'g_afs_oprit': 'D000037',      # Afstand tot oprit hoofdverkeersweg
        'g_afs_bieb': 'D000015',       # Afstand tot bibliotheek
    }
    
    # Human readable names for display
    INDICATOR_NAMES = {
        'g_afs_trein': 'Afstand tot treinstation',
        'g_afs_overstap': 'Afstand tot overstapstation',
        'g_afs_oprit': 'Afstand tot snelweg-oprit',
        'g_afs_bieb': 'Afstand tot bibliotheek',
    }
    
    # Unit (all in km)
    UNIT = 'km'
    
    def __init__(self, data_dir: str = 'data/raw/nabijheid'):
        """
        Initialize loader
        
        Args:
            data_dir: Directory containing yearly nabijheid data
        """
        self.data_dir = Path(data_dir)
        self.cache = {}
    
    def load_year(self, year: int) -> Optional[pd.DataFrame]:
        """
        Load nabijheid data for a specific year
        
        Args:
            year: Year to load (typically 2024 for most recent)
            
        Returns:
            DataFrame with proximity data, or None if not available
        """
        # Check cache
        if year in self.cache:
            return self.cache[year].copy()
        
        # Find Observations.csv file
        year_dir = self.data_dir / str(year)
        obs_file = year_dir / 'Observations.csv'
        
        if not obs_file.exists():
            print(f"⚠️  Nabijheid data {year} niet gevonden: {obs_file}")
            return None
        
        # Load CSV
        try:
            # CBS data uses semicolon separator
            df = pd.read_csv(obs_file, sep=';', encoding='utf-8', low_memory=False)
            
            # Expected columns
            required_cols = ['Measure', 'WijkenEnBuurten', 'Value']
            
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Verwachte kolommen niet gevonden in {obs_file.name}")
                print(f"    Gevonden: {df.columns.tolist()}")
                print(f"    Verwacht: {required_cols}")
                return None
            
            # Filter: only keep the indicators we want
            indicator_codes = list(self.INDICATORS.values())
            df = df[df['Measure'].isin(indicator_codes)].copy()
            
            # Rename columns for consistency
            df = df.rename(columns={
                'WijkenEnBuurten': 'gwb_code_10',
                'Measure': 'indicator_code',
                'Value': 'value'
            })
            
            # Filter: keep all levels (GM, WK, BU)
            df = df[df['gwb_code_10'].str.match(r'^(GM|WK|BU)', na=False)].copy()
            
            print(f"✅ Geladen: {obs_file.name}")
            print(f"   - Regio's: {df['gwb_code_10'].nunique():,}")
            print(f"   - Indicatoren: {df['indicator_code'].nunique()}")
            print(f"   - Totaal rijen: {len(df):,}")
            
            # Cache
            self.cache[year] = df
            
            return df.copy()
            
        except Exception as e:
            print(f"❌ Error loading {obs_file}: {e}")
            return None
    
    def pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot from long format (one row per indicator) to wide format (one row per region)
        
        Args:
            df: Long format DataFrame from load_year()
            
        Returns:
            Wide format DataFrame with proximity indicators as columns
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Convert values to float (handle European decimal format with comma)
        def convert_to_float(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, str):
                # Remove spaces and convert comma to dot
                val = val.strip().replace(',', '.')
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan
        
        df['value_float'] = df['value'].apply(convert_to_float)
        
        # Create pivot with indicators as columns
        pivot = df.pivot_table(
            index='gwb_code_10',
            columns='indicator_code',
            values='value_float',
            aggfunc='first'  # Should be unique anyway
        )
        
        # Reset index to make gwb_code_10 a column
        pivot = pivot.reset_index()
        
        # Rename indicator codes to short names (D000052 -> g_afs_trein)
        rename_map = {v: k for k, v in self.INDICATORS.items()}
        
        # Only rename columns that exist
        existing_renames = {old: new for old, new in rename_map.items() if old in pivot.columns}
        pivot = pivot.rename(columns=existing_renames)
        
        return pivot
    
    def load_and_process(self, year: int) -> Optional[pd.DataFrame]:
        """
        Load and fully process nabijheid data for a year
        
        Args:
            year: Year to load
            
        Returns:
            Processed DataFrame ready for merging with main data
        """
        # Load raw data
        df = self.load_year(year)
        if df is None:
            return None
        
        # Pivot to wide format
        df_wide = self.pivot_to_wide(df)
        
        if df_wide.empty:
            print(f"⚠️  Geen data na pivot voor jaar {year}")
            return None
        
        # Show summary
        print(f"\n📊 Nabijheid indicators geladen:")
        for short_name, full_name in self.INDICATOR_NAMES.items():
            if short_name in df_wide.columns:
                valid_count = df_wide[short_name].notna().sum()
                total_count = len(df_wide)
                coverage = (valid_count / total_count * 100) if total_count > 0 else 0
                mean_val = df_wide[short_name].mean()
                print(f"   ✅ {short_name:15s}: {valid_count:>5}/{total_count:>5} ({coverage:>5.1f}%) - mean: {mean_val:.2f} km")
            else:
                print(f"   ❌ {short_name:15s}: niet gevonden")
        
        return df_wide
    
    def get_available_years(self) -> List[int]:
        """
        Check which years have data available
        
        Returns:
            List of available years
        """
        available = []
        if self.data_dir.exists():
            for year_dir in self.data_dir.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    year = int(year_dir.name)
                    obs_file = year_dir / 'Observations.csv'
                    if obs_file.exists():
                        available.append(year)
        return sorted(available)


def load_nabijheid_data(
    year: int,
    data_dir: str = 'data/raw/nabijheid'
) -> Optional[pd.DataFrame]:
    """
    Convenience function to load and process nabijheid data
    
    Args:
        year: Year to load (typically 2024)
        data_dir: Directory with nabijheid data
        
    Returns:
        Processed DataFrame with proximity indicators, or None if not available
        
    Example:
        >>> nabijheid_df = load_nabijheid_data(2024)
        >>> print(nabijheid_df[['gwb_code_10', 'g_afs_trein', 'g_afs_bieb']])
    """
    loader = NabijheidLoader(data_dir=data_dir)
    return loader.load_and_process(year)


# Example usage and testing
if __name__ == '__main__':
    print("🚆 Nabijheid Loader Test\n")
    
    # Check available years
    loader = NabijheidLoader()
    available = loader.get_available_years()
    
    print(f"📊 Beschikbare jaren: {available}")
    
    if not available:
        print("\n⚠️  Geen nabijheid data gevonden!")
        print(f"    Verwachte locatie: data/raw/nabijheid/YYYY/Observations.csv")
    else:
        # Test load most recent year
        year = max(available)
        print(f"\n🔄 Test load jaar {year}...")
        
        df = loader.load_and_process(year)
        if df is not None:
            print(f"\n✅ Succesvol geladen:")
            print(f"   - Regio's: {len(df):,}")
            print(f"   - Indicatoren: {len([c for c in df.columns if c.startswith('g_afs_')])}")
            
            # Show sample
            print(f"\n📊 Sample (eerste 5 rijen):")
            cols_to_show = ['gwb_code_10'] + [c for c in df.columns if c.startswith('g_afs_')]
            print(df[cols_to_show].head(5).to_string(index=False))
            
            # Show statistics
            print(f"\n📏 Afstand statistieken:")
            for col in df.columns:
                if col.startswith('g_afs_'):
                    stats = df[col].describe()
                    print(f"\n   {col}:")
                    print(f"      Min:    {stats['min']:.2f} km")
                    print(f"      Mean:   {stats['mean']:.2f} km")
                    print(f"      Median: {stats['50%']:.2f} km")
                    print(f"      Max:    {stats['max']:.2f} km")
