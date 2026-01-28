"""
CBS-specific utilities voor data cleaning en validatie
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def clean_cbs_column(series: pd.Series) -> pd.Series:
    """
    Clean CBS column with '.' missing value indicators
    
    CBS quirks:
    - '.' means missing/unavailable
    - Comma used for decimals: "2,5" instead of "2.5"
    
    Args:
        series: Pandas Series to clean
        
    Returns:
        Cleaned Series with numeric values and NaN for missing
    """
    # Replace CBS missing indicators (suppress FutureWarning)
    cleaned = series.replace(['.', '', ' ', '..'], np.nan)
    
    # Explicitly infer objects to avoid FutureWarning
    if cleaned.dtype == 'object':
        cleaned = cleaned.infer_objects(copy=False)
    
    # If still object/string type, convert to numeric
    if cleaned.dtype == 'object':
        # Replace comma with dot for decimals
        cleaned = cleaned.astype(str).str.replace(',', '.')
        # Convert to numeric
        cleaned = pd.to_numeric(cleaned, errors='coerce')
    
    return cleaned


def clean_cbs_dataframe(df: pd.DataFrame, 
                        exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean all metric columns in dataframe
    
    Args:
        df: DataFrame to clean
        exclude_cols: List of columns to exclude from cleaning (geographic/text columns)
        
    Returns:
        Cleaned DataFrame
    """
    if exclude_cols is None:
        # Don't clean geographic/text columns
        exclude_cols = ['gwb_code_10', 'gwb_code_8', 'gwb_code',
                       'regio', 'gm_naam', 'recs']
    
    df = df.copy()
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = clean_cbs_column(df[col])
    
    return df


def identify_geo_level(code: str) -> str:
    """
    Identify geographic level from CBS code
    
    Args:
        code: CBS geographic code
        
    Returns:
        Geographic level: 'land', 'gemeente', 'wijk', or 'buurt'
    """
    if pd.isna(code):
        return 'unknown'
    
    code = str(code)
    
    if code == 'NL00':
        return 'land'
    elif code.startswith('GM'):
        return 'gemeente'
    elif code.startswith('WK'):
        return 'wijk'
    elif code.startswith('BU'):
        return 'buurt'
    else:
        # Fallback: check length
        if len(code) == 4:
            return 'land'
        elif len(code) == 8:
            return 'gemeente'
        elif len(code) == 10:
            return 'wijk'
        elif len(code) == 12:
            return 'buurt'
    
    return 'unknown'


def get_validated_columns():
    """
    Get validated column mappings for CBS KWB 2023 data
    
    Returns:
        Dictionary with column mappings per criterium
    """
    return {
        'geo_cols': [
            'gwb_code_10',  # Primary key
            'gwb_code_8',
            'regio',
            'gm_naam',
            'recs'
        ],
        'welvaart': {
            'g_ink_pi': 'Gemiddeld inkomen per inwoner',
            'p_ink_hi': '% Hoog inkomen',
            'p_ink_li': '% Laag inkomen (INVERSE)',
            'p_arb_pp': 'Participatiegraad',
            'g_wozbag': 'WOZ waarde x1000 euro',
            'p_koopw': '% Koopwoningen',
        },
        'family': {
            'a_inw': 'Aantal inwoners',
            'a_00_14': 'Aantal kinderen 0-14',
            'a_hh': 'Aantal huishoudens',
            'a_hh_m_k': 'Huishoudens met kinderen',
            'g_hhgro': 'Gem. huishoudgrootte',
            'g_afs_sc': 'Afstand school km (INVERSE)',
            'g_afs_kv': 'Afstand kdv km (INVERSE)',
            # Herkomst (absolute aantallen - we berekenen percentages)
            'a_nl_all': 'Aantal Nederlandse herkomst',
            'a_eur_al': 'Aantal Europese herkomst (excl NL)',
            'a_neu_al': 'Aantal Buiten-Europese herkomst',
        },
        'space': {
            'a_opp_ha': 'Totaal oppervlak ha',
            'a_lan_ha': 'Land oppervlak ha',
            'a_wat_ha': 'Water oppervlak ha',
            'bev_dich': 'Bevolkingsdichtheid (INVERSE)',
        },
        'amenities': {
            'g_afs_hp': 'Afstand huisarts km (INVERSE)',
            'g_afs_gs': 'Afstand supermarkt km (INVERSE)',
            'g_afs_sc': 'Afstand school km (INVERSE)',
        }
    }
