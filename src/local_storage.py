"""
Browser Local Storage for User Preferences

Uses browser localStorage API via streamlit-js-eval for scalable, per-user storage.
Works in deployment without needing server-side persistence.
"""

import json
import streamlit as st
from typing import Dict, Optional

try:
    from streamlit_js_eval import streamlit_js_eval
    JS_EVAL_AVAILABLE = True
except ImportError:
    JS_EVAL_AVAILABLE = False
    streamlit_js_eval = None


def save_to_local_storage(key: str, value: dict):
    """
    Save data to browser localStorage
    
    Args:
        key: Storage key (e.g., 'custom_weights')
        value: Dictionary to store
    """
    if not JS_EVAL_AVAILABLE:
        return
    
    # Convert dict to JSON string
    json_str = json.dumps(value)
    
    # JavaScript to save to localStorage
    js_code = f"""
    localStorage.setItem('{key}', '{json_str}');
    console.log('Saved to localStorage:', '{key}');
    """
    
    try:
        streamlit_js_eval(js_expressions=js_code, key=f'save_{key}')
    except Exception as e:
        # Silently fail if JS eval doesn't work
        pass


def load_from_local_storage(key: str, default: dict = None) -> Optional[dict]:
    """
    Load data from browser localStorage
    
    Args:
        key: Storage key (e.g., 'custom_weights')
        default: Default value if key doesn't exist
        
    Returns:
        Stored dictionary or default
        
    Note: streamlit_js_eval is async, so first call returns None.
    We need to trigger a rerun to get the actual value.
    """
    if default is None:
        default = {}
    
    if not JS_EVAL_AVAILABLE:
        return default
    
    # Cache keys
    cache_key = f'_loaded_{key}'
    loaded_flag = f'_localstorage_loaded_{key}'
    
    # If we already have loaded data, return it
    if loaded_flag in st.session_state and st.session_state[loaded_flag]:
        return st.session_state.get(cache_key, default)
    
    try:
        # JavaScript to load from localStorage
        js_code = f"localStorage.getItem('{key}')"
        result = streamlit_js_eval(js_expressions=js_code, key=f'load_{key}')
        
        # streamlit_js_eval returns None on first call (async)
        if result is None:
            # First call - JS is executing, will have value on rerun
            # Return default for now
            return default
        
        # Second call - we have the result
        if result:
            try:
                loaded_data = json.loads(result)
                st.session_state[cache_key] = loaded_data
                st.session_state[loaded_flag] = True
                return loaded_data
            except json.JSONDecodeError:
                # Invalid JSON in localStorage
                st.session_state[cache_key] = default
                st.session_state[loaded_flag] = True
                return default
        else:
            # localStorage is empty or returned empty string
            st.session_state[cache_key] = default
            st.session_state[loaded_flag] = True
            return default
            
    except Exception as e:
        # Silently fail if JS eval doesn't work
        st.session_state[cache_key] = default
        st.session_state[loaded_flag] = True
        return default


def clear_local_storage(key: str):
    """
    Clear data from browser localStorage
    
    Args:
        key: Storage key to clear
    """
    if not JS_EVAL_AVAILABLE:
        return
    
    js_code = f"""
    localStorage.removeItem('{key}');
    console.log('Cleared from localStorage:', '{key}');
    """
    
    try:
        streamlit_js_eval(js_expressions=js_code, key=f'clear_{key}')
    except Exception:
        pass
    
    # Also clear from session state cache
    cache_key = f'_loaded_{key}'
    if cache_key in st.session_state:
        del st.session_state[cache_key]


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
