"""
Trend Analysis Module

Calculate trends (growth/decline) for indicators across multiple years of KWB data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TrendAnalyzer:
    """Analyze trends across multiple years of CBS KWB data"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize trend analyzer
        
        Args:
            data_dir: Directory containing kwb-YYYY.xlsx files
        """
        self.data_dir = Path(data_dir)
        self.available_years = self._find_available_years()
        
    def _find_available_years(self) -> List[int]:
        """Find all available KWB data years"""
        years = []
        
        # Check both root data dir and data/raw/kwb subdirectories
        patterns = [
            self.data_dir / "kwb-*.xlsx",  # Root level
            self.data_dir / "raw" / "kwb" / "*" / "kwb-*.xlsx"  # Subdirectories
        ]
        
        for pattern in patterns:
            for file in self.data_dir.glob(str(pattern.relative_to(self.data_dir))):
                try:
                    # Extract year from filename kwb-YYYY.xlsx
                    year = int(file.stem.split("-")[1])
                    if year not in years:
                        years.append(year)
                except (IndexError, ValueError):
                    continue
        
        return sorted(years)
    
    def load_year_data(self, year: int, indicators: List[str]) -> pd.DataFrame:
        """
        Load specific indicators for a given year
        
        Args:
            year: Year to load
            indicators: List of indicator column names
            
        Returns:
            DataFrame with gwb_code_10 and indicator columns
        """
        # Try multiple possible paths
        possible_paths = [
            self.data_dir / f"kwb-{year}.xlsx",  # Root level
            self.data_dir / "raw" / "kwb" / str(year) / f"kwb-{year}.xlsx"  # Subdirectory
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
        
        if file_path is None:
            raise FileNotFoundError(f"KWB data for {year} not found in: {possible_paths}")
        
        # Load only needed columns for efficiency
        columns_to_load = ['gwb_code_10'] + indicators
        
        try:
            df = pd.read_excel(file_path, usecols=columns_to_load)
            df['year'] = year
            return df
        except Exception as e:
            print(f"Warning: Could not load {year} data from {file_path}: {e}")
            return pd.DataFrame()
    
    def calculate_trends(
        self, 
        indicator: str, 
        years: Optional[List[int]] = None,
        min_years: int = 2
    ) -> pd.DataFrame:
        """
        Calculate trend for a specific indicator across years
        
        Args:
            indicator: Indicator column name
            years: List of years to analyze (default: all available)
            min_years: Minimum years needed to calculate trend
            
        Returns:
            DataFrame with columns:
                - gwb_code_10
                - trend_absolute: absolute change (last - first)
                - trend_percentage: percentage change
                - trend_direction: 'stijgend', 'dalend', 'stabiel'
                - first_year_value
                - last_year_value
                - years_analyzed
        """
        if years is None:
            years = self.available_years
        
        if len(years) < min_years:
            raise ValueError(f"Need at least {min_years} years for trend analysis")
        
        # Load data for all years
        dfs = []
        for year in years:
            try:
                df_year = self.load_year_data(year, [indicator])
                if not df_year.empty:
                    dfs.append(df_year)
            except FileNotFoundError:
                continue
        
        if len(dfs) < min_years:
            raise ValueError(f"Could not load data for at least {min_years} years")
        
        # Combine all years
        df_all = pd.concat(dfs, ignore_index=True)
        
        # Calculate trends per location
        trends = []
        
        for code in df_all['gwb_code_10'].unique():
            df_location = df_all[df_all['gwb_code_10'] == code].copy()
            df_location = df_location.sort_values('year')
            
            # Need at least min_years data points
            if len(df_location) < min_years:
                continue
            
            # Get first and last non-null values
            valid_data = df_location[df_location[indicator].notna()]
            if len(valid_data) < min_years:
                continue
            
            first_row = valid_data.iloc[0]
            last_row = valid_data.iloc[-1]
            
            first_value = first_row[indicator]
            last_value = last_row[indicator]
            
            # Calculate trend
            absolute_change = last_value - first_value
            
            if first_value != 0:
                percentage_change = (absolute_change / first_value) * 100
            else:
                percentage_change = np.nan
            
            # Classify trend (threshold: 5% change is "significant")
            if abs(percentage_change) < 5:
                direction = 'stabiel'
            elif percentage_change > 0:
                direction = 'stijgend'
            else:
                direction = 'dalend'
            
            trends.append({
                'gwb_code_10': code,
                'trend_absolute': absolute_change,
                'trend_percentage': percentage_change,
                'trend_direction': direction,
                'first_year': first_row['year'],
                'last_year': last_row['year'],
                'first_year_value': first_value,
                'last_year_value': last_value,
                'years_analyzed': len(valid_data)
            })
        
        return pd.DataFrame(trends)
    
    def calculate_multi_indicator_trends(
        self,
        indicators: List[str],
        years: Optional[List[int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate trends for multiple indicators
        
        Args:
            indicators: List of indicator names
            years: Years to analyze
            
        Returns:
            Dict mapping indicator name to trend DataFrame
        """
        trends = {}
        
        for indicator in indicators:
            try:
                trend_df = self.calculate_trends(indicator, years)
                trends[indicator] = trend_df
                print(f"✓ Calculated trend for {indicator}: {len(trend_df)} locations")
            except Exception as e:
                print(f"✗ Could not calculate trend for {indicator}: {e}")
        
        return trends
    
    def merge_trends_with_current_data(
        self,
        current_df: pd.DataFrame,
        trend_dfs: Dict[str, pd.DataFrame],
        prefix: str = "trend_"
    ) -> pd.DataFrame:
        """
        Merge trend data with current dataset
        
        Args:
            current_df: Current data (e.g., from main_data.csv)
            trend_dfs: Dict of trend DataFrames
            prefix: Prefix for trend columns
            
        Returns:
            DataFrame with trend columns added
        """
        df_merged = current_df.copy()
        
        for indicator, trend_df in trend_dfs.items():
            # Rename columns with prefix
            trend_cols = {
                'trend_absolute': f'{prefix}{indicator}_abs',
                'trend_percentage': f'{prefix}{indicator}_pct',
                'trend_direction': f'{prefix}{indicator}_dir'
            }
            
            trend_df_renamed = trend_df[['gwb_code_10'] + list(trend_cols.keys())].copy()
            trend_df_renamed = trend_df_renamed.rename(columns=trend_cols)
            
            # Merge
            df_merged = df_merged.merge(
                trend_df_renamed,
                on='gwb_code_10',
                how='left'
            )
        
        return df_merged


def create_trend_score(
    df: pd.DataFrame,
    trend_indicators: List[str],
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Create a composite "trend score" based on multiple indicator trends
    
    Positive score = growing/improving area
    Negative score = declining area
    
    Args:
        df: DataFrame with trend columns (trend_<indicator>_pct)
        trend_indicators: List of indicator base names
        weights: Optional weights per indicator (default: equal weight)
        
    Returns:
        Series with trend scores
    """
    if weights is None:
        weights = {ind: 1.0 for ind in trend_indicators}
    
    score = pd.Series(0.0, index=df.index)
    total_weight = 0
    
    for indicator, weight in weights.items():
        col = f'trend_{indicator}_pct'
        if col in df.columns:
            # Normalize percentage changes to -100 to +100 scale
            normalized = df[col].clip(-100, 100)
            score += normalized * weight
            total_weight += weight
    
    if total_weight > 0:
        score = score / total_weight
    
    return score


# Convenience function
def analyze_trends_for_dashboard(
    data_dir: str = "data",
    indicators: List[str] = None,
    years: Optional[List[int]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to analyze trends for dashboard
    
    Args:
        data_dir: Data directory
        indicators: Indicators to analyze (default: key indicators)
        years: Years to use (default: all available)
        
    Returns:
        Dict of trend DataFrames
    """
    if indicators is None:
        # Default: key indicators for dashboard
        indicators = [
            'bev_dich',  # Bevolkingsdichtheid (indirect: groei = meer mensen)
            'pct_children',  # Kinderen percentage
            'pct_families',  # Gezinnen
            'ses_overall',  # SES development
            'g_ink_pi',  # Inkomen
            'p_arb_pp',  # Participatie
            'groen_percentage'  # Groen (kan dalen bij bebouwing)
        ]
    
    analyzer = TrendAnalyzer(data_dir)
    print(f"Found KWB data for years: {analyzer.available_years}")
    
    trends = analyzer.calculate_multi_indicator_trends(indicators, years)
    
    return trends


if __name__ == '__main__':
    # Test/demo
    print("=== Trend Analyzer Demo ===\n")
    
    # Initialize
    analyzer = TrendAnalyzer()
    print(f"Available years: {analyzer.available_years}\n")
    
    # Analyze key indicators
    test_indicators = ['pct_children', 'bev_dich', 'ses_overall']
    
    print(f"Analyzing trends for: {test_indicators}\n")
    trends = analyzer.calculate_multi_indicator_trends(test_indicators)
    
    # Show sample results
    for indicator, trend_df in trends.items():
        print(f"\n{indicator.upper()}:")
        print(f"  Total locations: {len(trend_df)}")
        
        if len(trend_df) > 0:
            # Direction summary
            direction_counts = trend_df['trend_direction'].value_counts()
            print(f"  Directions: {direction_counts.to_dict()}")
            
            # Top 5 growing
            top_growing = trend_df.nlargest(5, 'trend_percentage')
            print(f"\n  Top 5 stijgend:")
            for _, row in top_growing.iterrows():
                print(f"    {row['gwb_code_10']}: {row['trend_percentage']:.1f}% "
                      f"({row['first_year_value']:.1f} → {row['last_year_value']:.1f})")
            
            # Top 5 declining
            top_declining = trend_df.nsmallest(5, 'trend_percentage')
            print(f"\n  Top 5 dalend:")
            for _, row in top_declining.iterrows():
                print(f"    {row['gwb_code_10']}: {row['trend_percentage']:.1f}% "
                      f"({row['first_year_value']:.1f} → {row['last_year_value']:.1f})")
