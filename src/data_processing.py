"""
Data processing pipeline for CBS KWB data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.cbs_utils import clean_cbs_dataframe, get_validated_columns, identify_geo_level


class DataProcessor:
    """Process CBS KWB data: load, clean, and calculate derived metrics"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
    def load_data(self, year: int = 2023) -> pd.DataFrame:
        """
        Load CBS KWB data for specified year
        
        Args:
            year: Year to load (2022-2025, default 2023)
            
        Returns:
            Raw DataFrame
        """
        # Try new structure first (data/raw/kwb/YEAR/)
        filepath_new = self.data_dir / 'raw' / 'kwb' / str(year) / f'kwb-{year}.xlsx'
        # Fallback to old structure (data/)
        filepath_old = self.data_dir / f'kwb-{year}.xlsx'
        
        filepath = filepath_new if filepath_new.exists() else filepath_old
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath_new} or {filepath_old}")
        
        print(f"Loading {filepath}...")
        df = pd.read_excel(filepath)
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean CBS data (handle '.' missing values, convert types)
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning CBS data...")
        df_clean = clean_cbs_dataframe(df)
        
        # Add geographic level
        if 'gwb_code_10' in df_clean.columns:
            df_clean['geo_level'] = df_clean['gwb_code_10'].apply(identify_geo_level)
        
        print("Cleaning complete")
        return df_clean
    
    def select_relevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only relevant columns for dashboard
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with selected columns
        """
        cols_config = get_validated_columns()
        
        # Collect all relevant columns (use set to avoid duplicates)
        relevant_cols_set = set(cols_config['geo_cols'])
        
        for criterion in ['welvaart', 'family', 'space', 'amenities']:
            relevant_cols_set.update(cols_config[criterion].keys())
        
        # Convert back to list and add geo_level if it exists
        relevant_cols = list(relevant_cols_set)
        if 'geo_level' in df.columns:
            relevant_cols.append('geo_level')
        
        # Select only columns that exist
        existing_cols = [col for col in relevant_cols if col in df.columns]
        
        print(f"Selected {len(existing_cols)} columns")
        return df[existing_cols]
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics not directly in CBS data
        
        Args:
            df: DataFrame with base metrics
            
        Returns:
            DataFrame with derived metrics added
        """
        print("Calculating derived metrics...")
        df = df.copy()
        
        # Percentage children (0-14 years)
        if 'a_00_14' in df.columns and 'a_inw' in df.columns:
            # Avoid division by zero
            df['pct_children'] = np.where(
                df['a_inw'] > 0,
                (df['a_00_14'] / df['a_inw']) * 100,
                np.nan
            )
        
        # Percentage families with children
        if 'a_hh_m_k' in df.columns and 'a_hh' in df.columns:
            df['pct_families'] = np.where(
                df['a_hh'] > 0,
                (df['a_hh_m_k'] / df['a_hh']) * 100,
                np.nan
            )
        
        # Area per person (m²)
        if 'a_opp_ha' in df.columns and 'a_inw' in df.columns:
            # 1 hectare = 10,000 m²
            df['area_per_person_m2'] = np.where(
                df['a_inw'] > 0,
                (df['a_opp_ha'] / df['a_inw']) * 10000,
                np.nan
            )
        
        # Percentage water
        if 'a_wat_ha' in df.columns and 'a_opp_ha' in df.columns:
            df['pct_water'] = np.where(
                df['a_opp_ha'] > 0,
                (df['a_wat_ha'] / df['a_opp_ha']) * 100,
                np.nan
            )
        
        # Herkomst percentages
        if 'a_nl_all' in df.columns and 'a_inw' in df.columns:
            df['p_herk_nl'] = np.where(
                df['a_inw'] > 0,
                (df['a_nl_all'] / df['a_inw']) * 100,
                np.nan
            )
        
        if 'a_eur_al' in df.columns and 'a_inw' in df.columns:
            df['p_herk_eur'] = np.where(
                df['a_inw'] > 0,
                (df['a_eur_al'] / df['a_inw']) * 100,
                np.nan
            )
        
        if 'a_neu_al' in df.columns and 'a_inw' in df.columns:
            df['p_herk_neu'] = np.where(
                df['a_inw'] > 0,
                (df['a_neu_al'] / df['a_inw']) * 100,
                np.nan
            )
        
        derived_metrics = ['pct_children', 'pct_families', 'area_per_person_m2', 'pct_water', 
                          'p_herk_nl', 'p_herk_eur', 'p_herk_neu']
        print(f"Derived metrics calculated: {derived_metrics}")
        return df
    
    def process_year(self, year: int = 2023, save: bool = True) -> pd.DataFrame:
        """
        Complete processing pipeline for one year
        
        Args:
            year: Year to process
            save: Whether to save processed data
            
        Returns:
            Processed DataFrame
        """
        # Load
        df = self.load_data(year)
        
        # Clean
        df = self.clean_data(df)
        
        # Select relevant columns
        df = self.select_relevant_columns(df)
        
        # Calculate derived metrics
        df = self.calculate_derived_metrics(df)
        
        # Save if requested
        if save:
            output_path = self.processed_dir / f'kwb_{year}_cleaned.csv'
            df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
        
        return df


def main():
    """Test data processing"""
    processor = DataProcessor()
    
    # Process 2023 data
    df = processor.process_year(year=2023, save=True)
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"\nGeographic levels:")
    if 'geo_level' in df.columns:
        print(df['geo_level'].value_counts())
    
    print(f"\nSample data:")
    print(df.head())
    
    print(f"\nData completeness (% non-null):")
    completeness = (df.notna().sum() / len(df) * 100).sort_values(ascending=False)
    print(completeness.head(20))


if __name__ == '__main__':
    main()
