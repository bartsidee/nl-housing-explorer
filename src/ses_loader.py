"""
SES (Sociaaleconomische Status) data loader
Laadt CBS 85900NED data met percentielgroepen voor welvaart, inkomen en onderwijs
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class SESDataLoader:
    """Load and process SES (Socio-Economic Status) data from CBS 85900NED"""
    
    def __init__(self, data_dir: str = 'data/raw/ses/85900NED'):
        self.data_dir = Path(data_dir)
        
    def load_ses_data(self, year: int = 2022) -> pd.DataFrame:
        """
        Load SES data for specified year
        
        Args:
            year: Year to load (format: 2023 -> '2023JJ00')
            
        Returns:
            DataFrame with SES scores per wijk/buurt
        """
        # Load observations
        obs_path = self.data_dir / 'Observations.csv'
        obs_df = pd.read_csv(obs_path, sep=';')
        
        # Load measure codes (to understand what each measure means)
        measures_path = self.data_dir / 'MeasureCodes.csv'
        measures_df = pd.read_csv(measures_path, sep=';')
        
        # Load region codes
        regions_path = self.data_dir / 'WijkenEnBuurtenCodes.csv'
        regions_df = pd.read_csv(regions_path, sep=';')
        
        # Filter for specific year
        year_code = f'{year}JJ00'
        obs_filtered = obs_df[obs_df['Perioden'] == year_code].copy()
        
        print(f"Loaded SES data for {year}: {len(obs_filtered)} observations")
        
        # Convert Value column (replace comma with dot)
        obs_filtered['Value_numeric'] = obs_filtered['Value'].astype(str).str.replace(',', '.')
        obs_filtered['Value_numeric'] = pd.to_numeric(obs_filtered['Value_numeric'], errors='coerce')
        
        # Pivot to wide format
        ses_wide = obs_filtered.pivot_table(
            index='WijkenEnBuurten',
            columns='Measure',
            values='Value_numeric',
            aggfunc='first'
        ).reset_index()
        
        # Merge with region names
        ses_wide = ses_wide.merge(
            regions_df[['Identifier', 'Title', 'DimensionGroupId']],
            left_on='WijkenEnBuurten',
            right_on='Identifier',
            how='left'
        )
        
        # Extract key SES indicators
        ses_indicators = self._extract_ses_indicators(ses_wide, measures_df)
        
        return ses_indicators
    
    def _extract_ses_indicators(self, df: pd.DataFrame, measures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and calculate SES indicators
        
        Key measures:
        - D007966_1: Gemiddelde percentielgroep - Financi\u00eble welvaart
        - D007966_2: Gemiddelde percentielgroep - Inkomen
        - D007966_3: Gemiddelde percentielgroep - Onderwijsniveau
        """
        ses_df = pd.DataFrame()
        
        # Geographic identifiers
        ses_df['gwb_code_10'] = df['WijkenEnBuurten']
        ses_df['regio_naam'] = df['Title']
        ses_df['parent_code'] = df['DimensionGroupId']
        
        # SES indicators - Gemiddelde percentielgroepen (1-100 schaal)
        if 'D007966_1' in df.columns:
            ses_df['ses_welvaart'] = df['D007966_1']  # Financi\u00eble welvaart
        
        if 'D007966_2' in df.columns:
            ses_df['ses_inkomen'] = df['D007966_2']  # Inkomen
        
        if 'D007966_3' in df.columns:
            ses_df['ses_onderwijs'] = df['D007966_3']  # Onderwijsniveau
        
        # Calculate composite SES score (average of available indicators)
        ses_cols = [col for col in ['ses_welvaart', 'ses_inkomen', 'ses_onderwijs'] if col in ses_df.columns]
        if ses_cols:
            ses_df['ses_overall'] = ses_df[ses_cols].mean(axis=1)
        
        # Distribution percentages (optional, for detailed analysis)
        # Low SES (1-40e percentiel)
        if 'D007963_1' in df.columns:
            ses_df['pct_welvaart_laag'] = df['D007963_1']
        if 'D007963_2' in df.columns:
            ses_df['pct_inkomen_laag'] = df['D007963_2']
        if 'D007963_3' in df.columns:
            ses_df['pct_onderwijs_laag'] = df['D007963_3']
        
        # High SES (81-100e percentiel)
        if 'D007965_1' in df.columns:
            ses_df['pct_welvaart_hoog'] = df['D007965_1']
        if 'D007965_2' in df.columns:
            ses_df['pct_inkomen_hoog'] = df['D007965_2']
        if 'D007965_3' in df.columns:
            ses_df['pct_onderwijs_hoog'] = df['D007965_3']
        
        # Add geo level indicator
        ses_df['geo_level'] = ses_df['gwb_code_10'].apply(self._identify_geo_level)
        
        print(f"Extracted {len(ses_df)} SES records")
        print(f"Columns: {ses_df.columns.tolist()}")
        
        return ses_df
    
    def _identify_geo_level(self, code: str) -> str:
        """Identify geographic level from code"""
        if pd.isna(code):
            return 'unknown'
        code = str(code)
        if code.startswith('GM'):
            return 'gemeente'
        elif code.startswith('WK'):
            return 'wijk'
        elif code.startswith('BU'):
            return 'buurt'
        return 'unknown'
    
    def merge_with_kwb_data(self, kwb_df: pd.DataFrame, ses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge SES data with existing KWB data
        
        Args:
            kwb_df: Existing KWB DataFrame
            ses_df: SES DataFrame
            
        Returns:
            Merged DataFrame
        """
        # Merge on gwb_code_10
        merged = kwb_df.merge(
            ses_df,
            on='gwb_code_10',
            how='left',
            suffixes=('', '_ses')
        )
        
        print(f"Merged data: {len(merged)} rows")
        print(f"SES coverage: {merged['ses_overall'].notna().sum()} / {len(merged)} ({merged['ses_overall'].notna().sum()/len(merged)*100:.1f}%)")
        
        return merged


def main():
    """Test SES data loading"""
    loader = SESDataLoader()
    
    # Load 2022 SES data (most recent available)
    ses_df = loader.load_ses_data(year=2022)
    
    print("\n=== SES Data Summary ===")
    print(f"Total records: {len(ses_df)}")
    print(f"\nGeo levels:")
    print(ses_df['geo_level'].value_counts())
    
    print(f"\nSES Scores (percentielgroep 1-100):")
    for col in ['ses_welvaart', 'ses_inkomen', 'ses_onderwijs', 'ses_overall']:
        if col in ses_df.columns:
            valid_count = ses_df[col].notna().sum()
            print(f"  {col}: {ses_df[col].min():.1f} - {ses_df[col].max():.1f} (mean: {ses_df[col].mean():.1f}, n={valid_count})")
    
    print(f"\nSample data (top 10 by SES overall):")
    print(ses_df.nlargest(10, 'ses_overall')[['gwb_code_10', 'regio_naam', 'ses_welvaart', 'ses_inkomen', 'ses_onderwijs', 'ses_overall']])
    
    # Save
    output_path = Path('data/processed/ses_2022.csv')
    ses_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
