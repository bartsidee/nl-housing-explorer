"""
URL-based Weight Sharing

Export and import custom score weights via URL query parameters.
No localStorage - session state only.
"""

import json
import streamlit as st
from typing import Dict, Optional


def export_to_url(weights: Dict[str, float]) -> str:
    """
    Export weights to URL query parameters for sharing
    
    Args:
        weights: Dictionary of {indicator: weight}
        
    Returns:
        URL with encoded weights
    """
    import urllib.parse
    import base64
    
    # Compress weights (only non-zero values)
    compact_weights = {k: v for k, v in weights.items() if v != 0}
    
    # Encode as JSON then base64 for URL safety
    json_str = json.dumps(compact_weights)
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    
    return f"?weights={encoded}"


def import_from_url() -> Optional[Dict[str, float]]:
    """
    Import weights from URL query parameters
    
    Returns:
        Dictionary of weights or None if not found
    """
    import urllib.parse
    import base64
    
    # Get query params
    query_params = st.query_params
    
    if 'weights' in query_params:
        try:
            encoded = query_params['weights']
            # Decode base64 then JSON
            json_str = base64.urlsafe_b64decode(encoded.encode()).decode()
            weights = json.loads(json_str)
            return weights
        except Exception as e:
            st.error(f"Could not load weights from URL: {e}")
            return None
    
    return None
