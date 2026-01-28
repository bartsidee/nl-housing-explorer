"""
URL-based Configuration Sharing

Export and import custom score weights and filters via URL query parameters.
"""

import json
import streamlit as st
from typing import Dict, Optional


def export_to_url(weights: Dict[str, float], provincie: Optional[str] = None, 
                  gemeente: Optional[str] = None, kaart_niveau: Optional[str] = None) -> str:
    """
    Export weights and filters to URL query parameters for sharing
    
    Args:
        weights: Dictionary of {indicator: weight}
        provincie: Selected province (optional)
        gemeente: Selected gemeente (optional)
        kaart_niveau: Map level - gemeente/wijk/buurt (optional)
        
    Returns:
        URL with encoded parameters
    """
    import base64
    
    # Compress weights (only non-zero values)
    compact_weights = {k: v for k, v in weights.items() if v != 0}
    
    # Build config object
    config = {'weights': compact_weights}
    
    # Add filters if they're not default values
    if provincie and provincie != 'Alle Provincies':
        config['provincie'] = provincie
    
    if gemeente and gemeente != 'Heel Nederland':
        config['gemeente'] = gemeente
    
    if kaart_niveau and kaart_niveau != 'gemeente':
        config['kaart_niveau'] = kaart_niveau
    
    # Encode as JSON then base64 for URL safety
    json_str = json.dumps(config)
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    
    return f"?config={encoded}"


def import_from_url() -> Optional[Dict]:
    """
    Import weights and filters from URL query parameters
    
    Returns:
        Dictionary with 'weights', 'provincie', 'gemeente', 'kaart_niveau' or None if not found
    """
    import base64
    
    query_params = st.query_params
    
    if 'config' in query_params:
        try:
            encoded = query_params['config']
            # Decode base64 then JSON
            json_str = base64.urlsafe_b64decode(encoded.encode()).decode()
            config = json.loads(json_str)
            return config
        except Exception as e:
            st.error(f"Could not load configuration from URL: {e}")
            return None
    
    return None
