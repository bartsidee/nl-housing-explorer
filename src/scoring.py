"""
Scoring system for woonlocatie criteria
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.cbs_utils import get_validated_columns


class ScoringEngine:
    """Calculate normalized scores for woonlocatie criteria"""
    
    def __init__(self):
        self.columns_config = get_validated_columns()
        
    def normalize_column(self, series: pd.Series, inverse: bool = False) -> pd.Series:
        """
        Normalize column to 0-100 scale
        
        Args:
            series: Column to normalize
            inverse: If True, invert scale (lower values = higher score)
                    Use for distances and negative indicators
                    
        Returns:
            Normalized series (0-100)
        """
        # Replace inf/-inf with NaN first
        series_clean = series.replace([np.inf, -np.inf], np.nan)
        
        # Drop NaN values for scaling
        valid_mask = series_clean.notna()
        valid_count = valid_mask.sum()
        
        # Convert to Python int to avoid comparison issues
        if not isinstance(valid_count, (int, np.integer)):
            valid_count = int(valid_count)
        
        if valid_count == 0:
            return pd.Series(np.nan, index=series.index)
        
        values = series_clean[valid_mask].values.reshape(-1, 1)
        
        # Check if all values are the same
        val_min = float(values.min())
        val_max = float(values.max())
        
        if val_min == val_max:
            # All same value: assign 50 (middle score)
            normalized = pd.Series(50.0, index=series.index)
        else:
            # Scale to 0-100
            scaler = MinMaxScaler(feature_range=(0, 100))
            scaled_values = scaler.fit_transform(values).flatten()
            
            # Create result series with NaN for missing values
            normalized = pd.Series(np.nan, index=series.index)
            normalized.loc[valid_mask] = scaled_values
        
        # Inverse if needed (lower = better)
        if inverse:
            normalized = 100 - normalized
        
        return normalized
    
    def calculate_welvaart_score(self, df: pd.DataFrame, weights: Optional[Dict] = None) -> pd.Series:
        """
        Calculate Welvaart & Woning score
        
        Components:
        - Income (higher = better)
        - WOZ value (higher = better)
        - Low income % (lower = better, INVERSE)
        - Participation (higher = better)
        - Owner percentage (higher = better)
        """
        if weights is None:
            weights = {
                'g_ink_pi': 0.30,      # Average income
                'g_wozbag': 0.25,      # WOZ value
                'p_ink_li': 0.20,      # Low income % (inverse)
                'p_arb_pp': 0.15,      # Participation
                'p_koopw': 0.10,       # Owner %
            }
        
        scores = pd.DataFrame(index=df.index)
        
        # Normalize each component
        if 'g_ink_pi' in df.columns:
            scores['income'] = self.normalize_column(df['g_ink_pi'], inverse=False) * weights['g_ink_pi']
        
        if 'g_wozbag' in df.columns:
            scores['woz'] = self.normalize_column(df['g_wozbag'], inverse=False) * weights['g_wozbag']
        
        if 'p_ink_li' in df.columns:
            scores['low_income'] = self.normalize_column(df['p_ink_li'], inverse=True) * weights['p_ink_li']
        
        if 'p_arb_pp' in df.columns:
            scores['participation'] = self.normalize_column(df['p_arb_pp'], inverse=False) * weights['p_arb_pp']
        
        if 'p_koopw' in df.columns:
            scores['owner'] = self.normalize_column(df['p_koopw'], inverse=False) * weights['p_koopw']
        
        # Sum weighted scores
        total_score = scores.sum(axis=1)
        
        return total_score
    
    def calculate_family_score(self, df: pd.DataFrame, weights: Optional[Dict] = None) -> pd.Series:
        """
        Calculate Gezinsvriendelijk score
        
        Components:
        - % Children (higher = better)
        - % Families with children (higher = better)
        - Household size (higher = better)
        - Distance to school (lower = better, INVERSE)
        - Distance to daycare (lower = better, INVERSE)
        """
        if weights is None:
            weights = {
                'pct_children': 0.30,
                'pct_families': 0.30,
                'g_hhgro': 0.20,
                'g_afs_sc': 0.10,
                'g_afs_kv': 0.10,
            }
        
        scores = pd.DataFrame(index=df.index)
        
        if 'pct_children' in df.columns:
            scores['children'] = self.normalize_column(df['pct_children'], inverse=False) * weights['pct_children']
        
        if 'pct_families' in df.columns:
            scores['families'] = self.normalize_column(df['pct_families'], inverse=False) * weights['pct_families']
        
        if 'g_hhgro' in df.columns:
            scores['household_size'] = self.normalize_column(df['g_hhgro'], inverse=False) * weights['g_hhgro']
        
        if 'g_afs_sc' in df.columns:
            scores['school_dist'] = self.normalize_column(df['g_afs_sc'], inverse=True) * weights['g_afs_sc']
        
        if 'g_afs_kv' in df.columns:
            scores['daycare_dist'] = self.normalize_column(df['g_afs_kv'], inverse=True) * weights['g_afs_kv']
        
        total_score = scores.sum(axis=1)
        return total_score
    
    def calculate_space_score(self, df: pd.DataFrame, weights: Optional[Dict] = None) -> pd.Series:
        """
        Calculate Ruimte & Natuur score
        
        Components:
        - Area per person (higher = better)
        - % Water (higher = better)
        - Population density (lower = better, INVERSE)
        """
        if weights is None:
            weights = {
                'area_per_person_m2': 0.40,
                'pct_water': 0.30,
                'bev_dich': 0.30,
            }
        
        scores = pd.DataFrame(index=df.index)
        
        if 'area_per_person_m2' in df.columns:
            scores['area_pp'] = self.normalize_column(df['area_per_person_m2'], inverse=False) * weights['area_per_person_m2']
        
        if 'pct_water' in df.columns:
            scores['water'] = self.normalize_column(df['pct_water'], inverse=False) * weights['pct_water']
        
        if 'bev_dich' in df.columns:
            scores['density'] = self.normalize_column(df['bev_dich'], inverse=True) * weights['bev_dich']
        
        total_score = scores.sum(axis=1)
        return total_score
    
    def calculate_amenities_score(self, df: pd.DataFrame, weights: Optional[Dict] = None) -> pd.Series:
        """
        Calculate Bereikbaarheid score
        
        Components (all distances - lower = better, INVERSE):
        - Distance to GP
        - Distance to supermarket
        - Distance to school
        """
        if weights is None:
            weights = {
                'g_afs_hp': 0.40,   # GP
                'g_afs_gs': 0.35,   # Supermarket
                'g_afs_sc': 0.25,   # School
            }
        
        scores = pd.DataFrame(index=df.index)
        
        if 'g_afs_hp' in df.columns:
            scores['gp'] = self.normalize_column(df['g_afs_hp'], inverse=True) * weights['g_afs_hp']
        
        if 'g_afs_gs' in df.columns:
            scores['supermarket'] = self.normalize_column(df['g_afs_gs'], inverse=True) * weights['g_afs_gs']
        
        if 'g_afs_sc' in df.columns:
            scores['school'] = self.normalize_column(df['g_afs_sc'], inverse=True) * weights['g_afs_sc']
        
        total_score = scores.sum(axis=1)
        return total_score
    
    def calculate_overall_score(self, df: pd.DataFrame, 
                               criteria_weights: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate all criteria scores and overall score
        
        Args:
            df: DataFrame with processed data
            criteria_weights: Weights for each criterium (welvaart, family, space, amenities)
            
        Returns:
            DataFrame with all scores added
        """
        if criteria_weights is None:
            criteria_weights = {
                'welvaart': 0.25,
                'family': 0.25,
                'space': 0.25,
                'amenities': 0.25,
            }
        
        df = df.copy()
        
        print("Calculating scores...")
        
        # Calculate each criterium score
        df['welvaart_score'] = self.calculate_welvaart_score(df)
        df['family_score'] = self.calculate_family_score(df)
        df['space_score'] = self.calculate_space_score(df)
        df['amenities_score'] = self.calculate_amenities_score(df)
        
        # Calculate overall score
        df['overall_score'] = (
            df['welvaart_score'] * criteria_weights['welvaart'] +
            df['family_score'] * criteria_weights['family'] +
            df['space_score'] * criteria_weights['space'] +
            df['amenities_score'] * criteria_weights['amenities']
        )
        
        print("Scores calculated")
        return df
    
    def add_rankings(self, df: pd.DataFrame, geo_level: Optional[str] = None) -> pd.DataFrame:
        """
        Add rankings for each score
        
        Args:
            df: DataFrame with scores
            geo_level: If specified, rank only within this geographic level
            
        Returns:
            DataFrame with rankings added
        """
        df = df.copy()
        
        if geo_level and 'geo_level' in df.columns:
            # Filter to specific level
            mask = df['geo_level'] == geo_level
            subset = df[mask]
        else:
            subset = df
        
        # Rank overall score (higher = better, so ascending=False)
        df.loc[subset.index, 'overall_rank'] = subset['overall_score'].rank(
            ascending=False, method='min', na_option='bottom'
        )
        
        # Rank each criterium
        for score_col in ['welvaart_score', 'family_score', 'space_score', 'amenities_score']:
            rank_col = score_col.replace('_score', '_rank')
            df.loc[subset.index, rank_col] = subset[score_col].rank(
                ascending=False, method='min', na_option='bottom'
            )
        
        return df


def main():
    """Test scoring system"""
    from data_processing import DataProcessor
    
    # Load processed data
    processor = DataProcessor()
    df = processor.process_year(year=2023, save=False)
    
    # Calculate scores
    scorer = ScoringEngine()
    df = scorer.calculate_overall_score(df)
    df = scorer.add_rankings(df)
    
    # Save scored data
    output_path = Path('data/processed/kwb_2023_scored.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved scored data to: {output_path}")
    
    # Print summary
    print("\n=== Scoring Summary ===")
    print("\nScore ranges:")
    for col in ['welvaart_score', 'family_score', 'space_score', 'amenities_score', 'overall_score']:
        if col in df.columns:
            print(f"{col}: {df[col].min():.1f} - {df[col].max():.1f}")
    
    print("\nTop 10 overall (all levels):")
    top10 = df.nlargest(10, 'overall_score')[
        ['regio', 'geo_level', 'overall_score', 'welvaart_score', 'family_score', 'space_score', 'amenities_score']
    ]
    print(top10)
    
    print("\nTop 10 gemeenten:")
    gemeenten = df[df['geo_level'] == 'gemeente']
    top10_gem = gemeenten.nlargest(10, 'overall_score')[
        ['gm_naam', 'overall_score', 'welvaart_score', 'family_score', 'space_score', 'amenities_score']
    ]
    print(top10_gem)


if __name__ == '__main__':
    main()
