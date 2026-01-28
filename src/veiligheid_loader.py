"""
Veiligheid Data Loader

Loads and processes crime statistics from Politie Open Data (data.politie.nl)
for integration with CBS KWB data.

Author: AI Assistant
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')


class VeiligheidLoader:
    """Load and process Politie crime data"""
    
    # Crime categories (CBS codes from API)
    CRIME_CATEGORIES = {
        'totaal': '0.0.0',  # Totaal misdrijven
        'inbraak': '1.1.1',  # Diefstal/inbraak woning
        'geweld': '1.4.5',  # Mishandeling
        'auto': '1.2.1',  # Diefstal uit/vanaf motorvoertuigen
        'straat': '1.4.6',  # Straatroof
        'fiets': '1.2.3',  # Diefstal van brom-, snor-, fietsen
        'vernieling': '1.5.1',  # Vernieling/beschadiging
    }
    
    # Human readable names for display
    CRIME_NAMES = {
        '0.0.0': 'Totaal misdrijven',
        '1.1.1': 'Diefstal/inbraak woning',
        '1.4.5': 'Mishandeling',
        '1.2.1': 'Diefstal uit/vanaf motorvoertuigen',
        '1.4.6': 'Straatroof',
        '1.2.3': 'Diefstal van brom-, snor-, fietsen',
        '1.5.1': 'Vernieling/beschadiging',
    }
    
    # Weights for composite safety score (higher = more serious)
    DEFAULT_WEIGHTS = {
        'inbraak': 5.0,      # Most serious for residential choice
        'geweld': 4.0,       # Very serious
        'straat': 3.0,       # Serious
        'auto': 2.0,         # Moderate
        'vernieling': 2.0,   # Moderate
        'fiets': 1.0,        # Common but less serious
    }
    
    def __init__(self, data_dir: str = 'data/raw/veiligheid'):
        """
        Initialize loader
        
        Args:
            data_dir: Directory containing yearly crime data
        """
        self.data_dir = Path(data_dir)
        self.cache = {}
    
    def load_year(self, year: int) -> Optional[pd.DataFrame]:
        """
        Load crime data for a specific year
        
        Args:
            year: Year to load (2020-2025)
            
        Returns:
            DataFrame with crime data, or None if not available
        """
        # Check cache
        if year in self.cache:
            return self.cache[year].copy()
        
        # Find file
        year_dir = self.data_dir / str(year)
        possible_files = [
            year_dir / f'politie_misdrijven_{year}.csv',
            year_dir / 'politie_misdrijven.csv',
            year_dir / f'47022NED_{year}.csv',
            year_dir / f'47022NED.csv',
        ]
        
        csv_file = None
        for f in possible_files:
            if f.exists():
                csv_file = f
                break
        
        if csv_file is None:
            print(f"⚠️  Veiligheid data {year} niet gevonden in {year_dir}")
            print(f"    Download via: python3 scripts/download_veiligheid_data.py")
            return None
        
        # Load CSV
        try:
            # Politie data uses semicolon separator
            df = pd.read_csv(csv_file, sep=';', encoding='utf-8')
            
            # Expected columns (Dutch names from Politie portal)
            required_cols = ['WijkenEnBuurten', 'SoortMisdrijf', 'GeregistreerdeMisdrijven']
            
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Verwachte kolommen niet gevonden in {csv_file.name}")
                print(f"    Gevonden: {df.columns.tolist()}")
                print(f"    Verwacht: {required_cols}")
                return None
            
            # Rename to English for consistency
            df = df.rename(columns={
                'WijkenEnBuurten': 'gwb_code',
                'SoortMisdrijf': 'crime_type',
                'Perioden': 'period',
                'GeregistreerdeMisdrijven': 'count'
            })
            
            # Clean crime type codes (remove spaces)
            df['crime_type'] = df['crime_type'].str.strip()
            
            # Filter: keep only BU level (buurten) and valid crime types
            df = df[df['gwb_code'].str.startswith('BU', na=False)].copy()
            df = df[df['crime_type'].isin(self.CRIME_CATEGORIES.values())].copy()
            
            # Convert count to numeric
            df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
            
            print(f"✅ Geladen: {csv_file.name} - {len(df)} rijen, {df['gwb_code'].nunique()} buurten")
            
            # Cache
            self.cache[year] = df
            
            return df.copy()
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            return None
    
    def pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot from long format (one row per crime type) to wide format (one row per buurt)
        
        Args:
            df: Long format DataFrame from load_year()
            
        Returns:
            Wide format DataFrame with crime counts as columns
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Create pivot with crime types as columns
        pivot = df.pivot_table(
            index='gwb_code',
            columns='crime_type',
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        
        # Reset index to make gwb_code a column
        pivot = pivot.reset_index()
        
        # Rename columns to short English names
        rename_map = {v: k for k, v in self.CRIME_CATEGORIES.items()}
        
        # Only rename columns that exist
        existing_renames = {old: new for old, new in rename_map.items() if old in pivot.columns}
        pivot = pivot.rename(columns=existing_renames)
        
        return pivot
    
    def calculate_safety_scores(
        self, 
        df: pd.DataFrame, 
        population_col: str = 'inwoners',
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate safety scores from crime counts
        
        Args:
            df: DataFrame with crime counts and population
            population_col: Column name for population count
            weights: Optional custom weights for composite score
            
        Returns:
            DataFrame with added safety score columns
        """
        if df is None or df.empty:
            return df
        
        result = df.copy()
        
        # Use default weights if not provided
        if weights is None:
            weights = self.DEFAULT_WEIGHTS
        
        # Calculate per 1000 inhabitants
        for category in self.CRIME_CATEGORIES.keys():
            if category in result.columns and population_col in result.columns:
                # Use intuitive column names: crime_rate, inbraak_rate, etc.
                if category == 'totaal':
                    col_name = 'crime_rate'
                elif category == 'inbraak':
                    col_name = 'inbraak_rate'
                else:
                    col_name = f'{category}_per_1000'
                
                # Avoid division by zero
                result[col_name] = np.where(
                    result[population_col] > 0,
                    (result[category] / result[population_col]) * 1000,
                    0
                )
        
        # Note: We only keep the raw metric (misdrijven per 1000 inwoners)
        # No derived scores - keep it simple and transparent
        
        return result
    
    def _normalize_inverse(self, series: pd.Series, clip_percentile: float = 99) -> pd.Series:
        """
        Normalize series to 0-100 scale (INVERSE: lower values = higher scores)
        
        Uses percentile ranking to make scores comparable across regions.
        
        Args:
            series: Series to normalize
            clip_percentile: Percentile to clip outliers (default 99)
            
        Returns:
            Normalized series (0-100)
        """
        # Remove NaN and inf
        clean = series.replace([np.inf, -np.inf], np.nan)
        
        if clean.isna().all() or (clean == 0).all():
            return pd.Series(50, index=series.index)  # Neutral score
        
        # Clip outliers
        upper_bound = clean.quantile(clip_percentile / 100)
        clipped = clean.clip(upper=upper_bound)
        
        # Min-max normalization
        min_val = clipped.min()
        max_val = clipped.max()
        
        if max_val == min_val:
            return pd.Series(50, index=series.index)  # All same = neutral
        
        # Normalize to 0-100, then INVERSE (100 - x)
        normalized = ((clipped - min_val) / (max_val - min_val)) * 100
        inverted = 100 - normalized
        
        return inverted
    
    def load_and_process(
        self, 
        year: int, 
        population_df: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load and fully process crime data for a year
        
        Args:
            year: Year to load
            population_df: Optional DataFrame with population data (must have gwb_code_10 and inwoners)
            weights: Optional custom weights for safety score
            
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
        
        # Match CBS codes (gwb_code from Politie = gwb_code_10 from CBS)
        df_wide = df_wide.rename(columns={'gwb_code': 'gwb_code_10'})
        
        # If population data provided, merge and calculate scores
        if population_df is not None:
            # Merge with population
            df_merged = population_df[['gwb_code_10', 'inwoners']].merge(
                df_wide,
                on='gwb_code_10',
                how='left'
            )
            
            # Fill missing crime data with 0
            crime_cols = [col for col in df_wide.columns if col != 'gwb_code_10']
            for col in crime_cols:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col].fillna(0).astype(int)
            
            # Calculate safety scores
            df_merged = self.calculate_safety_scores(df_merged, weights=weights)
            
            # Drop temporary inwoners column (will be in main data already)
            df_merged = df_merged.drop(columns=['inwoners'], errors='ignore')
            
            return df_merged
        else:
            # Return without population normalization
            return df_wide
    
    def get_available_years(self) -> List[int]:
        """
        Check which years have data available
        
        Returns:
            List of available years
        """
        available = []
        for year in range(2020, 2026):  # Check 2020-2025
            year_dir = self.data_dir / str(year)
            if year_dir.exists() and any(year_dir.glob('*.csv')):
                available.append(year)
        return available


def load_politie_crime_data(
    year: int, 
    population_df: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    data_dir: str = 'data/raw/veiligheid'
) -> Optional[pd.DataFrame]:
    """
    Convenience function to load and process crime data
    
    Args:
        year: Year to load (2020-2023)
        population_df: Optional DataFrame with population data
        weights: Optional custom weights for safety score
        data_dir: Directory with crime data
        
    Returns:
        Processed DataFrame with safety indicators, or None if not available
        
    Example:
        >>> # Simple load without population normalization
        >>> crime_df = load_politie_crime_data(2023)
        
        >>> # With population data for normalization
        >>> population = pd.DataFrame({
        ...     'gwb_code_10': ['BU00000001', 'BU00000002'],
        ...     'inwoners': [1000, 2000]
        ... })
        >>> crime_df = load_politie_crime_data(2023, population_df=population)
        >>> print(crime_df[['gwb_code_10', 'veiligheid_score']])
    """
    loader = VeiligheidLoader(data_dir=data_dir)
    return loader.load_and_process(year, population_df, weights)


# Example usage and testing
if __name__ == '__main__':
    print("🚨 Veiligheid Loader Test\n")
    
    # Check available years
    loader = VeiligheidLoader()
    available = loader.get_available_years()
    
    print(f"📊 Beschikbare jaren: {available}")
    
    if not available:
        print("\n⚠️  Geen veiligheid data gevonden!")
        print("    Download via: python3 scripts/download_veiligheid_data.py")
    else:
        # Test load most recent year
        year = max(available)
        print(f"\n🔄 Test load jaar {year}...")
        
        df = loader.load_year(year)
        if df is not None:
            print(f"\n✅ Succesvol geladen:")
            print(f"   - Buurten: {df['gwb_code'].nunique()}")
            print(f"   - Crime types: {df['crime_type'].nunique()}")
            print(f"   - Totaal rijen: {len(df)}")
            
            # Show sample crime types
            print(f"\n📋 Crime categorieën (top 10):")
            top_types = df.groupby('crime_type')['count'].sum().sort_values(ascending=False).head(10)
            for crime_type, count in top_types.items():
                print(f"   - {crime_type}: {count:,}")
            
            # Test pivot
            print(f"\n🔄 Test pivot naar wide format...")
            df_wide = loader.pivot_to_wide(df)
            print(f"✅ Wide format: {len(df_wide)} buurten, {len(df_wide.columns)-1} crime columns")
            
        # Show sample
        print(f"\n📊 Sample (eerste 3 buurten):")
        cols_to_show = ['gwb_code_10'] + [c for c in ['totaal', 'inbraak', 'geweld'] if c in df_wide.columns]
        if all(c in df_wide.columns for c in cols_to_show):
            print(df_wide[cols_to_show].head(3).to_string(index=False))
        else:
            print(df_wide.head(3).to_string(index=False))
