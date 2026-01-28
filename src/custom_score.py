"""
Custom Score Calculator

Calculate weighted custom scores from selected indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class CustomScoreCalculator:
    """Calculate custom weighted scores from indicators"""
    
    # Define which indicators should be inverted (lower = better)
    INVERSE_INDICATORS = {
        'g_afs_hp', 'g_afs_gs', 'g_afs_sc', 'g_afs_kv',  # Afstanden (lower = better)
        'g_afs_trein', 'g_afs_overstap', 'g_afs_oprit', 'g_afs_bieb',  # Transport afstanden
        'p_ink_li',  # Laag inkomen % (lower = better)
        'bev_dich',  # Bevolkingsdichtheid (lower = better voor sommigen)
    }
    
    # Available indicators for custom scoring
    AVAILABLE_INDICATORS = {
        # SES Scores
        'ses_overall': ('SES - Overall', 'percentiel', False),
        'ses_welvaart': ('SES - Welvaart', 'percentiel', False),
        'ses_inkomen': ('SES - Inkomen', 'percentiel', False),
        'ses_onderwijs': ('SES - Onderwijs', 'percentiel', False),
        
        # Welvaart
        'g_ink_pi': ('Gemiddeld inkomen', '€', False),
        'g_wozbag': ('WOZ waarde', '€', False),  # Use negative weight for affordability
        'p_arb_pp': ('Participatiegraad', '%', False),
        'p_ink_li': ('Huishoudens laag inkomen', '%', True),  # Inverse
        'p_ink_hi': ('Huishoudens hoog inkomen', '%', False),
        'p_koopw': ('Koopwoningen', '%', False),
        
        # Gezin
        'pct_children': ('Kinderen 0-14 jaar', '%', False),
        'pct_families': ('Gezinnen met kinderen', '%', False),
        'g_hhgro': ('Huishoudgrootte', 'personen', False),
        
        # Herkomst / Diversiteit
        'p_herk_nl': ('Nederlandse herkomst', '%', False),
        'p_herk_eur': ('Europese herkomst (excl NL)', '%', False),
        'p_herk_neu': ('Buiten-Europese herkomst', '%', False),
        
        # Voorzieningen (afstanden - lower is better)
        'g_afs_sc': ('Afstand basisschool', 'km', True),  # Inverse
        'g_afs_kv': ('Afstand kinderopvang', 'km', True),  # Inverse
        'g_afs_hp': ('Afstand huisarts', 'km', True),  # Inverse
        'g_afs_gs': ('Afstand supermarkt', 'km', True),  # Inverse
        
        # Transport & Bereikbaarheid (afstanden - lower is better)
        'g_afs_trein': ('Afstand treinstation', 'km', True),  # Inverse
        'g_afs_overstap': ('Afstand overstapstation', 'km', True),  # Inverse
        'g_afs_oprit': ('Afstand snelweg-oprit', 'km', True),  # Inverse
        'g_afs_bieb': ('Afstand bibliotheek', 'km', True),  # Inverse
        
        # Ruimte & Groen
        'area_per_person_m2': ('Oppervlakte per persoon', 'm²', False),
        'pct_water': ('Wateroppervlak', '%', False),
        'bev_dich': ('Bevolkingsdichtheid', '/km²', True),  # Inverse (optioneel)
        'a_opp_ha': ('Totale oppervlakte', 'ha', False),
        'groen_percentage': ('Groenpercentage', '%', False),  # RIVM 2022 data
        
        # Terrein & Klimaat
        'nap_hoogte_gem': ('NAP Hoogte gemiddeld', 'm', False),  # Higher elevation
        # Note: Can use positive weight for higher ground, negative for sea level
        
        # Veiligheid
        'crime_rate': ('Misdrijven (totaal)', 'per 1000 inw.', True),  # Inverse: lower crime = better
        'inbraak_rate': ('Inbraak', 'per 1000 inw.', True),  # Inverse: lower = better
    }
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    def normalize_indicator(
        self, 
        series: pd.Series, 
        inverse: bool = False
    ) -> pd.Series:
        """
        Normalize indicator to 0-100 scale using min-max normalization
        
        Args:
            series: Indicator values
            inverse: If True, invert scale (lower values = higher score)
            
        Returns:
            Normalized series (0-100 scale)
        """
        # Remove NaN values for min/max calculation
        valid_values = series.dropna()
        
        if len(valid_values) == 0:
            return pd.Series(np.nan, index=series.index)
        
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        # Avoid division by zero
        if max_val == min_val:
            return pd.Series(50.0, index=series.index)
        
        # Min-max normalization to 0-100
        normalized = ((series - min_val) / (max_val - min_val)) * 100
        
        # Invert if needed (for "lower is better" indicators)
        if inverse:
            normalized = 100 - normalized
        
        return normalized
    
    def calculate_custom_score(
        self,
        df: pd.DataFrame,
        indicator_weights: Dict[str, float]
    ) -> pd.Series:
        """
        Calculate weighted custom score from selected indicators
        
        Supports negative weights to reverse indicator direction per profile.
        Example: g_wozbag with weight +2.0 = prefer higher WOZ (wealthy areas)
                 g_wozbag with weight -2.0 = prefer lower WOZ (affordable areas)
        
        Args:
            df: DataFrame with indicator data
            indicator_weights: Dict of {indicator_name: weight} (can be negative)
            
        Returns:
            Series with custom scores (0-100 scale)
        """
        if not indicator_weights:
            # No indicators selected - return NaN
            return pd.Series(np.nan, index=df.index)
        
        # Normalize each indicator and apply weight
        weighted_scores = []
        total_weight = 0
        
        for indicator, weight in indicator_weights.items():
            if weight == 0:
                continue
            
            if indicator not in df.columns:
                continue
            
            # Check if indicator should be inverted (from AVAILABLE_INDICATORS)
            _, _, is_inverse_by_default = self.AVAILABLE_INDICATORS.get(
                indicator, 
                (None, None, False)
            )
            
            # For negative weights, flip the inversion logic
            # Example: g_wozbag is False by default (higher=better)
            #   - weight +2.0: use False (higher WOZ = better)
            #   - weight -2.0: use True (higher WOZ = worse, so invert to lower=better)
            is_inverse = is_inverse_by_default
            if weight < 0:
                is_inverse = not is_inverse_by_default
            
            # Normalize indicator
            normalized = self.normalize_indicator(df[indicator], inverse=is_inverse)
            
            # Apply absolute weight (direction already handled by inversion)
            weighted = normalized * abs(weight)
            weighted_scores.append(weighted)
            total_weight += abs(weight)
        
        if not weighted_scores or total_weight == 0:
            return pd.Series(np.nan, index=df.index)
        
        # Calculate weighted average
        score_sum = pd.concat(weighted_scores, axis=1).sum(axis=1)
        custom_score = score_sum / total_weight
        
        return custom_score
    
    def get_available_indicators(self) -> List[Tuple[str, str, str, bool]]:
        """
        Get list of available indicators for custom scoring
        
        Returns:
            List of (indicator_key, display_name, unit, is_inverse)
        """
        return [
            (key, name, unit, is_inverse)
            for key, (name, unit, is_inverse) in self.AVAILABLE_INDICATORS.items()
        ]
    
    def get_indicator_info(self, indicator_key: str) -> Tuple[str, str, bool]:
        """
        Get display info for an indicator
        
        Args:
            indicator_key: Indicator column name
            
        Returns:
            (display_name, unit, is_inverse)
        """
        if indicator_key in self.AVAILABLE_INDICATORS:
            return self.AVAILABLE_INDICATORS[indicator_key]
        return (indicator_key, '', False)


# Convenience function
def calculate_custom_score(
    df: pd.DataFrame,
    indicator_weights: Dict[str, float]
) -> pd.Series:
    """
    Calculate custom score - convenience wrapper
    
    Args:
        df: DataFrame with indicator data
        indicator_weights: Dict of {indicator_name: weight}
        
    Returns:
        Series with custom scores (0-100 scale)
    """
    calculator = CustomScoreCalculator()
    return calculator.calculate_custom_score(df, indicator_weights)
