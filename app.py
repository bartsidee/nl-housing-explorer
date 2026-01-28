"""
🏘️ NL Housing Explorer

Features:
- 22+ ruwe CBS indicatoren (geen gewogen scores)
- SES percentielgroepen (CBS 85900NED)
- Simpele gemeente filter met auto-zoom
- 🗺️ Interactive choropleth map (PDOK WFS)
- Complete data: 3,352 wijken, 14,421 buurten
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add components and src to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

# Import map component
try:
    from components.map_viewer import load_and_merge_geo_data, create_choropleth_map
    from streamlit_folium import st_folium
    from src.pdok_geo_loader import PDOKGeoLoader
    from src.provincie_mapping import get_provincie_for_gemeente, get_all_provincies, get_gemeenten_in_provincie
    from src.custom_score import CustomScoreCalculator, calculate_custom_score
    MAP_AVAILABLE = True
except ImportError as e:
    MAP_AVAILABLE = False
    print(f"Warning: Map components not available: {e}")


MAP_INDICATOR_BASE_OPTIONS = [
    ('🚨 Misdrijven - Totaal (per 1000 inw.)', 'crime_rate'),
    ('🏠 Inbraak (per 1000 inw.)', 'inbraak_rate'),
    ('SES - Overall (percentiel)', 'ses_overall'),
    ('SES - Welvaart (percentiel)', 'ses_welvaart'),
    ('SES - Inkomen (percentiel)', 'ses_inkomen'),
    ('SES - Onderwijs (percentiel)', 'ses_onderwijs'),
    ('Gemiddeld inkomen (€)', 'g_ink_pi'),
    ('WOZ waarde gemiddeld (€)', 'g_wozbag'),
    ('Participatiegraad (%)', 'p_arb_pp'),
    ('Huishoudens laag inkomen (%)', 'p_ink_li'),
    ('Huishoudens hoog inkomen (%)', 'p_ink_hi'),
    ('Koopwoningen (%)', 'p_koopw'),
    ('Kinderen 0-14 jaar (%)', 'pct_children'),
    ('Gezinnen met kinderen (%)', 'pct_families'),
    ('Gemiddelde huishoudgrootte (personen)', 'g_hhgro'),
    ('Afstand basisschool (km)', 'g_afs_sc'),
    ('Afstand kinderopvang (km)', 'g_afs_kv'),
    ('Oppervlakte per persoon (m²)', 'area_per_person_m2'),
    ('Wateroppervlak (%)', 'pct_water'),
    ('Bevolkingsdichtheid (per km²)', 'bev_dich'),
    ('Totale oppervlakte (ha)', 'a_opp_ha'),
    ('Afstand huisarts (km)', 'g_afs_hp'),
    ('Afstand supermarkt (km)', 'g_afs_gs'),
    ('Afstand treinstation (km) 🚆', 'g_afs_trein'),
    ('Afstand overstapstation (km) 🚉', 'g_afs_overstap'),
    ('Afstand snelweg-oprit (km) 🚗', 'g_afs_oprit'),
    ('Afstand bibliotheek (km) 📚', 'g_afs_bieb'),
    ('Groenpercentage (%) 🌳', 'groen_percentage'),
    ('NAP Hoogte gemiddeld (m) 🏔️', 'nap_hoogte_gem'),
    ('Nederlandse herkomst (%)', 'p_herk_nl'),
    ('Europese herkomst excl. NL (%)', 'p_herk_eur'),
    ('Buiten-Europese herkomst (%)', 'p_herk_neu'),
]
MAP_INDICATOR_LABEL_TO_COLUMN = {label: column for label, column in MAP_INDICATOR_BASE_OPTIONS}

# Page config
st.set_page_config(
    page_title="NL Housing Explorer",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .breadcrumb {
        font-size: 0.9rem;
        color: #666;
        padding: 0.5rem 0;
    }
    .ses-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .ses-high {
        background-color: #d4edda;
        color: #155724;
    }
    .ses-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .ses-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_with_ses():
    """Load processed data with trends (Parquet format)"""
    parquet_path = Path('data/processed/current/main_data_with_trends.parquet')
    
    if not parquet_path.exists():
        st.error("""
        ❌ Data file not found!
        
        Please run trend processing first:
        ```bash
        python3 scripts/process_multiyear_trends.py
        ```
        
        This will generate: data/processed/current/main_data_with_trends.parquet
        """)
        st.stop()
    
    df = pd.read_parquet(parquet_path)
    return df


def load_custom_weights_from_session():
    """
    Load custom score weights from:
    1. URL query params (for sharing)
    2. Session state (current session)
    """
    # Strategy 1: Check URL for shared weights (highest priority)
    try:
        from src.url_sharing import import_from_url
        url_weights = import_from_url()
        if url_weights:
            return url_weights
    except Exception:
        pass
    
    # Strategy 2: Check session state (current session)
    if '_persisted_weights' in st.session_state:
        return st.session_state['_persisted_weights']
    
    return {}


def get_available_presets():
    """Get list of available preset profiles"""
    presets = {}
    presets_dir = Path('data/presets')
    
    # Create presets directory if it doesn't exist
    presets_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all user_settings_example_*.json files
    for preset_file in presets_dir.glob('user_settings_example_*.json'):
        try:
            with open(preset_file, 'r') as f:
                data = json.load(f)
                preset_name = data.get('_preset', preset_file.stem.replace('user_settings_example_', '').title())
                preset_description = data.get('_description', '')
                presets[preset_name] = {
                    'file': preset_file,
                    'description': preset_description,
                    'weights': data.get('custom_weights', {})
                }
        except Exception as e:
            print(f"Warning: Could not load preset {preset_file}: {e}")
    
    return presets


def load_preset_weights(preset_weights):
    """Load preset weights into session state"""
    # Clear existing weights first
    calculator = CustomScoreCalculator()
    for key in calculator.AVAILABLE_INDICATORS.keys():
        st.session_state[f'weight_{key}'] = 0.0
    
    # Load preset weights (skip _ prefixed keys)
    for key, value in preset_weights.items():
        if not key.startswith('_'):
            try:
                st.session_state[f'weight_{key}'] = float(value)
            except (ValueError, TypeError):
                pass


def save_custom_weights_to_session(weights_dict):
    """
    Save custom score weights to session state
    
    Note: Weights persist only during the current session.
    Use 'Share Configuration' button to generate a URL for persistence.
    """
    # Save to session state (current session only)
    st.session_state['_persisted_weights'] = weights_dict


def get_ses_badge_html(ses_score):
    """Create SES badge HTML"""
    if pd.isna(ses_score):
        return '<span class="ses-badge">SES: N/A</span>'
    
    if ses_score >= 60:
        class_name = "ses-high"
        label = "Hoog"
    elif ses_score >= 40:
        class_name = "ses-medium"
        label = "Midden"
    else:
        class_name = "ses-low"
        label = "Laag"
    
    return f'<span class="ses-badge {class_name}">SES: {ses_score:.0f} ({label})</span>'


def create_ses_radar_chart(row):
    """Create radar chart including SES scores"""
    categories = ['Welvaart', 'Gezin', 'Ruimte', 'Voorzieningen']
    values = [
        row.get('welvaart_score', 0),
        row.get('family_score', 0),
        row.get('space_score', 0),
        row.get('amenities_score', 0)
    ]
    
    # Add SES if available
    if 'ses_overall' in row and pd.notna(row['ses_overall']):
        categories.append('SES')
        values.append(row['ses_overall'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Scores'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400
    )
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🏘️ NL Woonlocatie Verkenner</div>', unsafe_allow_html=True)
    st.markdown("**CBS Kerncijfers Wijken en Buurten:** 31 indicatoren | 6 databronnen | Trend analyse 2020-2025")
    
    # Load saved weights from URL/session
    saved_weights = load_custom_weights_from_session()
    
    # Only populate weight sliders if we haven't already AND we have data
    if 'weights_loaded_from_url' not in st.session_state:
        if saved_weights:
            # Pre-populate session state with saved weights (skip comment fields)
            for key, value in saved_weights.items():
                # Skip comment/instruction fields (start with _)
                if not key.startswith('_'):
                    try:
                        st.session_state[f'weight_{key}'] = float(value)
                    except (ValueError, TypeError):
                        pass  # Skip invalid values
        st.session_state['weights_loaded_from_url'] = True
    
    # Load data
    with st.spinner("Data laden..."):
        df = load_data_with_ses()
    
    # Calculate custom score early if weights exist in session state
    # This ensures custom_score is available for kaartfilter widget
    if 'custom_indicator_weights' in st.session_state and st.session_state['custom_indicator_weights']:
        early_weights = st.session_state['custom_indicator_weights']
        if early_weights:  # Extra check for non-empty dict
            df['custom_score'] = calculate_custom_score(df, early_weights)
    
    # Sidebar - Simplified with Provincie Filter!
    st.sidebar.header("🗺️ Locatie Filter")
    
    st.sidebar.subheader("📍 Selecteer Locatie")
    
    # Provincie filter (optional)
    all_provincies = ['Alle Provincies'] + get_all_provincies()
    selected_provincie = st.sidebar.selectbox(
        "Provincie (optioneel)",
        options=all_provincies,
        key='provincie_select',
        help="Filter gemeenten op provincie"
    )
    
    # Get gemeenten (filtered by provincie if selected)
    all_gemeenten = sorted(df[df['geo_level'] == 'gemeente']['gm_naam'].dropna().unique())
    
    if selected_provincie != 'Alle Provincies':
        # Filter gemeenten to only show those in selected provincie
        gemeenten_in_provincie = get_gemeenten_in_provincie(selected_provincie)
        gemeenten = sorted([g for g in all_gemeenten if g in gemeenten_in_provincie])
        st.sidebar.caption(f"{len(gemeenten)} gemeenten in {selected_provincie}")
    else:
        gemeenten = all_gemeenten
    
    # Gemeente selector
    gemeente_options = ['Heel Nederland'] + list(gemeenten)
    selected_gemeente = st.sidebar.selectbox(
        "Gemeente",
        options=gemeente_options,
        key='gemeente_select'
    )
    
    map_level_options = ['gemeente', 'wijk', 'buurt']
    map_geo_level = st.session_state.get('map_view_level', 'gemeente')
    indicator_filter_ranges = {}
    with st.sidebar.expander("📍 Kaartfilter", expanded=False):
        map_geo_level_index = map_level_options.index(map_geo_level) if map_geo_level in map_level_options else 0
        map_geo_level = st.radio(
            "Kaartniveau",
            options=map_level_options,
            index=map_geo_level_index,
            format_func=str.title,
            key='map_view_level'
        )
        st.caption("Selecteer indicatoren die alle weergegeven locaties moeten voldoen.")
        
        # Build filter options - include custom score if configured
        filter_options_list = list(MAP_INDICATOR_BASE_OPTIONS)
        if 'custom_indicator_weights' in st.session_state and st.session_state['custom_indicator_weights']:
            filter_options_list.insert(0, ('🎯 Mijn Custom Score', 'custom_score'))
        
        filter_options = [label for label, _ in filter_options_list]
        filter_label_to_column = {label: column for label, column in filter_options_list}
        
        selected_filters = st.multiselect(
            "Filter indicatoren",
            options=filter_options,
            key='map_indicator_filters'
        )
        for label in selected_filters:
            column = filter_label_to_column.get(label)
            if column is None:
                continue
            
            # For custom_score, ensure it exists in the dataframe
            if column == 'custom_score' and column not in df.columns:
                st.caption(f"⚠️ {label}: bereken eerst custom score in Tab 1")
                continue
            level_values = df[df['geo_level'] == map_geo_level][column].dropna()
            if level_values.empty:
                st.caption(f"Geen data voor {label} op niveau {map_geo_level}")
                continue
            min_val = float(level_values.min())
            max_val = float(level_values.max())
            if min_val == max_val:
                max_val = min_val + 1
            
            # Smart step size based on indicator type
            value_range = max_val - min_val
            if column == 'nap_hoogte_gem':
                # NAP height: 0.5m precision (people care about smaller differences)
                step = 0.5
                decimals = 1
            elif column in ['crime_rate', 'inbraak_rate']:
                # Crime rates: 0.1 per 1000 precision
                step = max(value_range / 100, 0.1)
                decimals = 1
            elif column in ['groen_percentage', 'pct_water', 'pct_children', 'pct_families', 
                           'p_koopw', 'p_arb_pp', 'p_ink_li', 'p_ink_hi']:
                # Percentages: 0.5% precision
                step = 0.5
                decimals = 1
            elif column.startswith('g_afs_'):
                # Distances: 0.1 km precision
                step = 0.1
                decimals = 1
            elif column in ['g_ink_pi', 'g_wozbag']:
                # Money: €1000 precision
                step = max(value_range / 50, 1000)
                decimals = 0
            elif column in ['ses_overall', 'ses_welvaart', 'ses_inkomen', 'ses_onderwijs']:
                # SES percentiles: 1 point precision
                step = 1.0
                decimals = 0
            elif column == 'bev_dich':
                # Population density: 10 per km² precision
                step = max(value_range / 100, 10)
                decimals = 0
            else:
                # Default: 40 steps across range, min 0.1
                step = max(value_range / 40, 0.1)
                decimals = 3
            
            slider_key = f"map_filter_{column}_{map_geo_level}"
            low, high = st.slider(
                label,
                min_value=round(min_val, decimals),
                max_value=round(max_val, decimals),
                value=(round(min_val, decimals), round(max_val, decimals)),
                step=step,
                key=slider_key
            )
            indicator_filter_ranges[column] = (low, high)

    selected_gemeente_code = None
    # Determine filtering based on selection
    if selected_gemeente != 'Heel Nederland':
        # Get gemeente code
        gemeente_data = df[(df['geo_level'] == 'gemeente') & (df['gm_naam'] == selected_gemeente)]
        if len(gemeente_data) > 0:
            selected_gemeente_code = gemeente_data['gemeente_code'].iloc[0]
            
            # Show provincie if known
            prov = get_provincie_for_gemeente(selected_gemeente)
            if prov:
                st.sidebar.markdown(f'**📍 {selected_gemeente}** ({prov})')
            else:
                st.sidebar.markdown(f'**📍 {selected_gemeente}**')
            
            # Filter to selected gemeente at all levels (gemeente/wijk/buurt)
            df_filtered = df[df['gemeente_code'] == selected_gemeente_code].copy()
            geo_level = 'gemeente'  # Default for table display
        else:
            st.sidebar.warning("Gemeente niet gevonden")
            df_filtered = df[df['geo_level'] == 'gemeente'].copy()
            geo_level = 'gemeente'
    else:
        # Show gemeenten (filtered by provincie if selected)
        if selected_provincie != 'Alle Provincies':
            # Filter to gemeenten in selected provincie
            df_filtered = df[
                (df['geo_level'] == 'gemeente') & 
                (df['gm_naam'].isin(gemeenten))
            ].copy()
        else:
            # Show all gemeenten in Nederland
            df_filtered = df[df['geo_level'] == 'gemeente'].copy()
        geo_level = 'gemeente'

    # Build map dataset based on desired level and filters
    map_df = df[df['geo_level'] == map_geo_level].copy()
    if selected_provincie != 'Alle Provincies':
        provincie_codes = df[
            (df['geo_level'] == 'gemeente') &
            (df['gm_naam'].isin(gemeenten))
        ]['gemeente_code'].dropna().unique()
        map_df = map_df[map_df['gemeente_code'].isin(provincie_codes)]
    if selected_gemeente != 'Heel Nederland' and selected_gemeente_code:
        map_df = map_df[map_df['gemeente_code'] == selected_gemeente_code]
    for col, (low, high) in indicator_filter_ranges.items():
        if col in map_df.columns:
            map_df = map_df[
                map_df[col].notna() &
                (map_df[col] >= low) &
                (map_df[col] <= high)
            ]
    
    st.sidebar.markdown(f"**Aantal locaties:** {len(df_filtered)}")
    
    # 🎯 CUSTOM SCORE CONFIGURATOR
    with st.sidebar.expander("🎯 Maak je eigen score", expanded=False):
        st.markdown("**Selecteer indicatoren en stel gewichten in**")
        st.caption("Hogere gewichten = belangrijker in je score (0.0 - 5.0)")
        
        # Show message if weights came from URL (shared configuration)
        query_params = st.query_params
        if 'weights' in query_params and not st.session_state.get('_url_weights_loaded', False):
            st.info("🔗 Gedeelde configuratie geladen vanuit URL!")
            st.session_state['_url_weights_loaded'] = True
        
        # Preset selector
        presets = get_available_presets()
        if presets:
            st.markdown("---")
            st.markdown("**📋 Laad een Profiel**")
            
            preset_options = ['- Kies een profiel -'] + list(presets.keys())
            selected_preset = st.selectbox(
                "Profiel",
                options=preset_options,
                key='preset_selector',
                label_visibility='collapsed'
            )
            
            if selected_preset != '- Kies een profiel -':
                preset_info = presets[selected_preset]
                st.caption(f"💡 {preset_info['description']}")
                
                col_load, col_clear = st.columns([1, 1])
                with col_load:
                    if st.button("✅ Laad Profiel", key='load_preset', use_container_width=True):
                        load_preset_weights(preset_info['weights'])
                        st.success(f"Profiel '{selected_preset}' geladen!")
                        st.rerun()
                with col_clear:
                    if st.button("🔄 Reset", key='reset_from_preset', use_container_width=True):
                        calculator_temp = CustomScoreCalculator()
                        
                        # Clear all weight values in session state
                        for key in calculator_temp.AVAILABLE_INDICATORS.keys():
                            st.session_state[f'weight_{key}'] = 0.0
                        
                        # Clear session state flags
                        if '_persisted_weights' in st.session_state:
                            del st.session_state['_persisted_weights']
                        
                        st.rerun()
        
        st.markdown("---")
        st.markdown("**⚙️ Pas Gewichten Aan**")
        st.caption("💡 Tip: Gebruik **negatieve gewichten** (bijv. -2.0) om richting om te draaien")
        st.caption("   Voorbeeld: WOZ +2.0 = voorkeur voor dure gebieden, WOZ -2.0 = betaalbaar")
        
        # Reset button
        if st.button("🔄 Reset Alles", help="Wis alle gewichten", key='reset_all_weights'):
            calculator_temp = CustomScoreCalculator()
            
            # Clear all weight values in session state
            for key in calculator_temp.AVAILABLE_INDICATORS.keys():
                if f'weight_{key}' in st.session_state:
                    st.session_state[f'weight_{key}'] = 0.0
            
            # Clear session state flags
            if '_persisted_weights' in st.session_state:
                del st.session_state['_persisted_weights']
            
            st.rerun()
        
        # Initialize calculator
        calculator = CustomScoreCalculator()
        available = calculator.get_available_indicators()
        
        # Group indicators by category
        categories = {
            'SES': [ind for ind in available if ind[0].startswith('ses_')],
            'Welvaart & Woning': [ind for ind in available if ind[0] in [
                'g_ink_pi', 'g_wozbag', 'p_arb_pp', 'p_ink_li', 'p_ink_hi', 'p_koopw'
            ]],
            'Gezin': [ind for ind in available if ind[0] in [
                'pct_children', 'pct_families', 'g_hhgro'
            ]],
            'Herkomst': [ind for ind in available if ind[0] in [
                'p_herk_nl', 'p_herk_eur', 'p_herk_neu'
            ]],
            'Veiligheid': [ind for ind in available if ind[0] in ['crime_rate', 'inbraak_rate']],
            'Voorzieningen': [ind for ind in available if ind[0].startswith('g_afs_')],
            'Ruimte & Groen': [ind for ind in available if ind[0] in [
                'area_per_person_m2', 'pct_water', 'bev_dich', 'a_opp_ha', 'groen_percentage'
            ]],
        }
        
        # Store selected indicators and weights
        indicator_weights = {}
        
        # First pass: collect all current weights from session state
        category_weights = {}
        for category, indicators in categories.items():
            category_sum = 0
            for ind_key, ind_name, ind_unit, is_inverse in indicators:
                weight_key = f"weight_{ind_key}"
                if weight_key in st.session_state:
                    weight_val = st.session_state[weight_key]
                    if weight_val != 0:
                        category_sum += abs(weight_val)  # Use absolute value for percentage
            category_weights[category] = category_sum
        
        # Calculate total weight for percentages
        total_weight_all = sum(category_weights.values())
        
        # Second pass: render UI with percentages
        for category, indicators in categories.items():
            if not indicators:
                continue
            
            # Calculate percentage for this category
            if total_weight_all > 0:
                category_pct = (category_weights[category] / total_weight_all) * 100
                st.markdown(f"**{category}** ({category_pct:.0f}%)")
            else:
                st.markdown(f"**{category}**")
            
            for ind_key, ind_name, ind_unit, is_inverse in indicators:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Indicator label with inverse indicator
                    label = f"{ind_name}"
                    if is_inverse:
                        label += " ↓"  # Arrow down = lower is better
                    st.caption(label)
                
                with col2:
                    # Weight input - use session state for value if available, else 0.0
                    weight_key = f"weight_{ind_key}"
                    if weight_key not in st.session_state:
                        st.session_state[weight_key] = 0.0
                    
                    # Weight input - supports negative weights for reversing indicators
                    weight = st.number_input(
                        f"w_{ind_key}",
                        min_value=-5.0,  # Negative weights = reverse direction
                        max_value=5.0,
                        value=st.session_state[weight_key],  # Explicit value prevents -5.0 default
                        step=0.1,
                        format="%.1f",
                        label_visibility="collapsed",
                        key=weight_key
                    )
                    
                    if weight != 0:
                        indicator_weights[ind_key] = weight
        
        # Show summary
        if indicator_weights:
            st.markdown("---")
            st.markdown(f"**📊 {len(indicator_weights)} indicatoren geselecteerd**")
            total_weight = sum(indicator_weights.values())
            st.caption(f"Totaal gewicht: {total_weight:.1f}")
            
            # Show top 3 weighted indicators
            sorted_weights = sorted(indicator_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            st.caption("Top 3:")
            for ind_key, weight in sorted_weights:
                ind_name, _, _ = calculator.get_indicator_info(ind_key)
                st.caption(f"  • {ind_name}: {weight:.1f}")
            
            # Save to session state for persistence
            save_custom_weights_to_session(indicator_weights)
            
            # Share configuration button
            st.markdown("---")
            if st.button("🔗 Deel Configuratie", help="Genereer een URL om je instellingen te delen", use_container_width=True):
                from src.url_sharing import export_to_url
                share_url_suffix = export_to_url(indicator_weights)
                
                # Get current page URL (works in both dev and production)
                try:
                    # Try to get actual URL from Streamlit config
                    base_url = st.get_option("browser.serverAddress")
                    if not base_url or base_url == "localhost":
                        # Fallback to localhost for development
                        base_url = "http://localhost:8501"
                except:
                    # Fallback
                    base_url = "http://localhost:8501"
                
                full_url = f"{base_url}{share_url_suffix}"
                
                st.code(full_url, language=None)
                st.caption("💾 Kopieer deze URL om je configuratie te delen!")
                st.caption("ℹ️ Werkt cross-device en met anderen")
        else:
            st.info("💡 Stel gewichten in om je custom score te activeren")
        
        # Store in session state for use in tabs
        st.session_state['custom_indicator_weights'] = indicator_weights
    
    # Calculate/update custom score if weights are set
    # IMPORTANT: Calculate on full df for consistent normalization across all tabs
    if indicator_weights:
        # Recalculate with potentially updated weights
        df['custom_score'] = calculate_custom_score(df, indicator_weights)
    
    # Merge custom_score into filtered dataframes
    if 'custom_score' in df.columns:
        # Merge into df_filtered
        df_filtered = df_filtered.copy()
        df_filtered = df_filtered.drop(columns=['custom_score'], errors='ignore')
        df_filtered = df_filtered.merge(
            df[['gwb_code_10', 'custom_score']], 
            on='gwb_code_10', 
            how='left'
        )
        # Merge into map_df for map visualization
        map_df = map_df.drop(columns=['custom_score'], errors='ignore')
        map_df = map_df.merge(
            df[['gwb_code_10', 'custom_score']], 
            on='gwb_code_10', 
            how='left'
        )
    else:
        # No custom score - set to NaN
        df_filtered['custom_score'] = np.nan
        map_df['custom_score'] = np.nan
    
    # === MAIN CONTENT ===
    
    # Tab persistence using URL query params
    query_params = st.query_params
    default_tab = 0
    if 'tab' in query_params:
        try:
            default_tab = int(query_params['tab'])
            if not (0 <= default_tab <= 3):
                default_tab = 0
        except (ValueError, TypeError):
            default_tab = 0
    
    # Store in session state
    if 'active_tab_index' not in st.session_state:
        st.session_state['active_tab_index'] = default_tab
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Kaart",
        "📊 Data Verkenning",
        "🔍 Locatie Details",
        "📈 Vergelijking"
    ])
    
    # TAB 1: Interactive Map - Simplified!
    with tab1:
        st.header("🗺️ Interactieve Kaart")
        
        if not MAP_AVAILABLE:
            st.error("Kaart component niet beschikbaar. Installeer: pip install folium streamlit-folium geopandas")
        else:
            # Indicator selector - ALLE ruwe CBS indicatoren
            map_options = list(MAP_INDICATOR_BASE_OPTIONS)
            
            # Add custom score if configured
            if 'custom_indicator_weights' in st.session_state and st.session_state['custom_indicator_weights']:
                map_options.insert(0, ('🎯 Mijn Custom Score', 'custom_score'))
            
            map_indicator = st.selectbox(
                "📊 Kies indicator om op kaart te visualiseren:",
                options=map_options,
                format_func=lambda x: x[0],
                index=0
            )
            
            indicator_col = map_indicator[1]
            indicator_name = map_indicator[0]
            
            # Show color scheme info
            from components.map_viewer import get_colorscheme_for_indicator, get_indicator_interpretation
            colorscheme = get_colorscheme_for_indicator(indicator_col)
            interpretation = get_indicator_interpretation(indicator_col)
            
            if 'RdYlGn' in colorscheme:
                st.info(f"ℹ️ **Kleurlegenda:** {interpretation} (Rood = ver/hoog = slecht)")
            elif colorscheme == 'YlGnBu':
                st.info(f"ℹ️ **Kleurlegenda:** {interpretation} (Geel = laag)")
            else:
                st.info(f"ℹ️ **Kleurlegenda:** {interpretation}")
            
            # Show data stats (using df_filtered from sidebar!)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🗺️ Locaties", len(map_df))
            with col_b:
                if len(map_df) > 0 and indicator_col in map_df.columns:
                    coverage = map_df[indicator_col].notna().sum()
                    st.metric("📊 Data coverage", f"{coverage/len(map_df)*100:.0f}%")
                else:
                    st.metric("📊 Data coverage", "0%")
            with col_c:
                if indicator_col in map_df.columns:
                    valid_values = map_df[indicator_col].dropna()
                    if not valid_values.empty:
                        st.metric("📈 Mediaan", f"{valid_values.median():.1f}")
            
            # Load and create map
            st.caption(f"**Kaart:** {indicator_name} | **Niveau:** {map_geo_level.title()}")
            
            try:
                with st.spinner("Kaart laden..."):
                    if len(map_df) == 0:
                        st.warning("Geen locaties voldoen aan de kaartfilters")
                    elif indicator_col not in map_df.columns:
                        st.warning(f"Indicator '{indicator_col}' niet beschikbaar in huidige dataset")
                    else:
                        # Load geo data for current level
                        geo_loader = PDOKGeoLoader()
                        if map_geo_level == 'gemeente':
                            gdf = geo_loader.get_gemeenten()
                        elif map_geo_level == 'wijk':
                            gdf = geo_loader.get_wijken()
                        else:
                            gdf = geo_loader.get_buurten()
                        
                        # Merge with filtered data
                        # Use 'inner' to show ONLY areas that match filters (filtered areas disappear)
                        gdf_with_data = gdf.merge(
                            map_df[['gwb_code_10', 'display_name', indicator_col]],
                            left_on='statcode',
                            right_on='gwb_code_10',
                            how='inner'
                        )
                        
                        if len(gdf_with_data) == 0:
                            st.warning("Geen geo-data beschikbaar")
                        else:
                            # Create map with auto-zoom and FIXED scale using full dataset
                            # This ensures consistent colors across all geographic filters
                            fig_map = create_choropleth_map(
                                gdf_with_data,
                                score_column=indicator_col,
                                auto_zoom=True,  # Always auto-zoom!
                                use_fixed_scale=True,  # NEW: Use fixed national baseline
                                baseline_data=df  # Full dataset for consistent color scale
                            )
                            
                            # Display map
                            st_folium(fig_map, width='stretch', height=600, returned_objects=[])
                            st.caption(f"✅ Toont {len(gdf_with_data)} locaties | Kleuren: vaste schaal o.b.v. landelijk gemiddelde")
            
            except Exception as e:
                st.error(f"Fout bij laden: {str(e)}")
    
    # TAB 2: Data Verkenning
    with tab2:
        st.header("📊 CBS Data Verkenning")
        
        st.markdown("""
        Bekijk en sorteer op **ruwe CBS indicatoren** - geen gewogen scores, directe data uit CBS Kerncijfers.
        """)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            top_n = st.number_input("Toon top", min_value=5, max_value=100, value=20, step=5)
        with col2:
            # Indicator options with units  
            sort_options = [
                ('🚨 Misdrijven - Totaal (per 1000 inw.)', 'crime_rate'),
                ('🏠 Inbraak (per 1000 inw.)', 'inbraak_rate'),
                ('SES - Overall (percentiel)', 'ses_overall'),
                ('SES - Welvaart (percentiel)', 'ses_welvaart'),
                ('SES - Inkomen (percentiel)', 'ses_inkomen'),
                ('SES - Onderwijs (percentiel)', 'ses_onderwijs'),
                ('Gemiddeld inkomen (€)', 'g_ink_pi'),
                ('WOZ waarde (€)', 'g_wozbag'),
                ('Participatiegraad (%)', 'p_arb_pp'),
                ('Laag inkomen (%)', 'p_ink_li'),
                ('Hoog inkomen (%)', 'p_ink_hi'),
                ('Koopwoningen (%)', 'p_koopw'),
                ('Kinderen 0-14 jaar (%)', 'pct_children'),
                ('Gezinnen met kinderen (%)', 'pct_families'),
                ('Huishoudgrootte (personen)', 'g_hhgro'),
                ('Afstand basisschool (km)', 'g_afs_sc'),
                ('Afstand kinderopvang (km)', 'g_afs_kv'),
                ('Afstand huisarts (km)', 'g_afs_hp'),
                ('Afstand supermarkt (km)', 'g_afs_gs'),
                ('Afstand treinstation (km)', 'g_afs_trein'),
                ('Afstand overstapstation (km)', 'g_afs_overstap'),
                ('Afstand snelweg-oprit (km)', 'g_afs_oprit'),
                ('Afstand bibliotheek (km)', 'g_afs_bieb'),
                ('Oppervlakte per persoon (m²)', 'area_per_person_m2'),
                ('Wateroppervlak (%)', 'pct_water'),
                ('Bevolkingsdichtheid (per km²)', 'bev_dich'),
                ('Totale oppervlakte (ha)', 'a_opp_ha'),
                ('Groenpercentage (%)', 'groen_percentage'),
                ('NAP Hoogte gemiddeld (m)', 'nap_hoogte_gem'),
                # Herkomst
                ('Herkomst: Nederland (%)', 'p_herk_nl'),
                ('Herkomst: Europa (excl. NL) (%)', 'p_herk_eur'),
                ('Herkomst: Buiten-Europa (%)', 'p_herk_neu'),
            ]
            
            # Add trend indicators if available
            trend_indicators_available = [col for col in df.columns if col.startswith('trend_') and col.endswith('_pct')]
            if 'trend_score' in df.columns:
                sort_options.insert(0, ('📈 Trend Score (dynamiek)', 'trend_score'))
            if 'trend_pct_children_pct' in df.columns:
                sort_options.insert(1 if 'trend_score' in df.columns else 0, 
                                  ('📈 Kinderen trend (%)', 'trend_pct_children_pct'))
            if 'trend_bev_dich_pct' in df.columns:
                sort_options.insert(2 if 'trend_score' in df.columns else 1,
                                  ('📈 Bevolkingsgroei (%)', 'trend_bev_dich_pct'))
            
            # Add custom score if configured
            if 'custom_indicator_weights' in st.session_state and st.session_state['custom_indicator_weights']:
                sort_options.insert(0, ('🎯 Mijn Custom Score', 'custom_score'))
            
            sort_indicator = st.selectbox(
                "Sorteer op indicator", 
                options=sort_options,
                format_func=lambda x: x[0]
            )
        
        sort_col = sort_indicator[1]
        sort_name = sort_indicator[0]
        
        # Determine sort direction (lower is better for some indicators)
        inverse_indicators = ['p_ink_li', 'bev_dich', 'g_afs_hp', 'g_afs_gs', 'g_afs_sc', 'g_afs_kv']
        ascending = sort_col in inverse_indicators
        
        # Get top locations (handle missing data)
        df_valid = df_filtered[df_filtered[sort_col].notna()]
        
        if len(df_valid) == 0:
            st.warning("Geen data beschikbaar voor deze indicator met huidige filters")
        else:
            top_df = df_valid.nlargest(top_n, sort_col) if not ascending else df_valid.nsmallest(top_n, sort_col)
            
            st.subheader(f"Top {len(top_df)} - {sort_name}")
            
            # Create display dataframe
            display_data = []
            for idx, row in top_df.iterrows():
                # Determine parent context based on geo_level
                parent_context = None
                if row['geo_level'] == 'gemeente':
                    # Show provincie
                    parent_context = get_provincie_for_gemeente(row['gm_naam'])
                elif row['geo_level'] == 'wijk':
                    # Show gemeente
                    parent_context = row['gm_naam']
                elif row['geo_level'] == 'buurt':
                    # Show wijk name
                    if 'wijk_code' in row and pd.notna(row['wijk_code']):
                        wijk_match = df[(df['geo_level'] == 'wijk') & (df['gwb_code_10'] == row['wijk_code'])]
                        if len(wijk_match) > 0:
                            parent_context = wijk_match.iloc[0]['regio_naam']
                
                data_row = {
                    'Rang': len(display_data) + 1,
                    'Locatie': row['display_name'],
                    'Niveau': row['geo_level'].title(),
                }
                
                # Add parent context column with appropriate label
                if row['geo_level'] == 'gemeente':
                    data_row['Provincie'] = parent_context if parent_context else 'N/A'
                elif row['geo_level'] == 'wijk':
                    data_row['Gemeente'] = parent_context if parent_context else 'N/A'
                elif row['geo_level'] == 'buurt':
                    data_row['Wijk'] = parent_context if parent_context else 'N/A'
                
                data_row['Waarde'] = row[sort_col]
                data_row['SES'] = row.get('ses_overall', None) if 'ses_overall' in df.columns else None
                
                # Add trend info if available (NEW!)
                trend_col = f'trend_{sort_col}_pct'
                trend_dir_col = f'trend_{sort_col}_dir'
                
                if trend_col in row.index and pd.notna(row[trend_col]):
                    trend_pct = row[trend_col]
                    trend_dir = row.get(trend_dir_col, '')
                    
                    # Arrow based on direction
                    if trend_dir == 'stijgend':
                        arrow = '↑'
                    elif trend_dir == 'dalend':
                        arrow = '↓'
                    else:
                        arrow = '↔'
                    
                    data_row['Trend'] = f"{arrow} {trend_pct:+.1f}%"
                
                display_data.append(data_row)
            
            display_df = pd.DataFrame(display_data)
            
            # Format based on indicator type
            if sort_col in ['g_ink_pi', 'g_wozbag']:
                # CBS data is in thousands (x1000)
                display_df['Waarde'] = display_df['Waarde'].apply(lambda x: f"€{x*1000:,.0f}" if pd.notna(x) else "N/A")
            elif 'pct_' in sort_col or sort_col.startswith('p_'):
                display_df['Waarde'] = display_df['Waarde'].apply(lambda x: f"{x:.1f}%")
            elif 'afs_' in sort_col:
                display_df['Waarde'] = display_df['Waarde'].apply(lambda x: f"{x:.1f} km")
            elif sort_col == 'area_per_person_m2':
                display_df['Waarde'] = display_df['Waarde'].apply(lambda x: f"{x:.0f} m²")
            elif sort_col == 'bev_dich':
                display_df['Waarde'] = display_df['Waarde'].apply(lambda x: f"{x:.0f}/km²")
            else:
                display_df['Waarde'] = display_df['Waarde'].apply(lambda x: f"{x:.1f}")
            
            # Format SES
            if 'SES' in display_df.columns:
                display_df['SES'] = display_df['SES'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
            
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            # Summary stats
            st.markdown("### 📈 Statistieken")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Hoogste", f"{df_valid[sort_col].max():.1f}")
            with col_b:
                st.metric("Gemiddelde", f"{df_valid[sort_col].mean():.1f}")
            with col_c:
                st.metric("Mediaan", f"{df_valid[sort_col].median():.1f}")
            with col_d:
                coverage = df_filtered[sort_col].notna().sum()
                st.metric("Data coverage", f"{coverage/len(df_filtered)*100:.0f}%")
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download als CSV",
                data=csv,
                file_name=f"top_{top_n}_{sort_col}_{geo_level}.csv",
                mime="text/csv"
            )
    
    # TAB 3: Locatie Details - Ruwe CBS Data
    with tab3:
        st.header("🔍 Locatie Details - CBS Kerncijfers")
        
        location_options = sorted(df_filtered['display_name'].dropna().unique())
        
        if len(location_options) == 0:
            st.warning("Geen locaties beschikbaar met de huidige filters. Pas de selectie aan.")
        else:
            selected_location = st.selectbox(
                f"Selecteer {geo_level}:",
                options=location_options
            )
            
            # Check if location exists in filtered data
            location_matches = df_filtered[df_filtered['display_name'] == selected_location]
            if len(location_matches) == 0:
                st.error(f"Locatie '{selected_location}' niet gevonden in gefilterde data.")
            else:
                location_data = location_matches.iloc[0]
                
                # Display location header
                st.markdown(f"### 📍 {selected_location}")
                st.caption(f"Niveau: {location_data['geo_level'].title()} | Code: {location_data['gwb_code_10']}")
                
                # Basic demographics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'a_inw' in location_data and pd.notna(location_data['a_inw']):
                        st.metric("👥 Inwoners", f"{int(location_data['a_inw']):,}")
                with col2:
                    if 'a_hh' in location_data and pd.notna(location_data['a_hh']):
                        st.metric("🏠 Huishoudens", f"{int(location_data['a_hh']):,}")
                with col3:
                    if 'a_opp_ha' in location_data and pd.notna(location_data['a_opp_ha']):
                        st.metric("📏 Oppervlakte", f"{int(location_data['a_opp_ha'])} ha")
                
                st.markdown("---")
                
                # SES Scores
                st.subheader("🎓 SES - Sociaaleconomische Status")
                ses_cols = st.columns(4)
                ses_indicators = [
                    ('Overall', 'ses_overall'),
                    ('Welvaart', 'ses_welvaart'),
                    ('Inkomen', 'ses_inkomen'),
                    ('Onderwijs', 'ses_onderwijs')
                ]
                for i, (name, col) in enumerate(ses_indicators):
                    with ses_cols[i]:
                        if col in location_data and pd.notna(location_data[col]):
                            st.metric(name, f"{location_data[col]:.0f}", help="Percentiel 1-100")
                        else:
                            st.metric(name, "N/A")
                
                st.markdown("---")
                
                # Inkomen & Welvaart
                st.subheader("💰 Inkomen & Welvaart")
                col1, col2 = st.columns(2)
                with col1:
                    if 'g_ink_pi' in location_data and pd.notna(location_data['g_ink_pi']):
                        st.metric("Gemiddeld inkomen", f"€{location_data['g_ink_pi']*1000:,.0f}")
                    if 'p_arb_pp' in location_data and pd.notna(location_data['p_arb_pp']):
                        st.metric("Participatiegraad", f"{location_data['p_arb_pp']:.1f}%")
                    if 'p_koopw' in location_data and pd.notna(location_data['p_koopw']):
                        st.metric("Koopwoningen", f"{location_data['p_koopw']:.1f}%")
                
                with col2:
                    if 'g_wozbag' in location_data and pd.notna(location_data['g_wozbag']):
                        st.metric("WOZ waarde gemiddeld", f"€{location_data['g_wozbag']*1000:,.0f}")
                    if 'p_ink_li' in location_data and pd.notna(location_data['p_ink_li']):
                        st.metric("Huishoudens laag inkomen", f"{location_data['p_ink_li']:.1f}%")
                    if 'p_ink_hi' in location_data and pd.notna(location_data['p_ink_hi']):
                        st.metric("Huishoudens hoog inkomen", f"{location_data['p_ink_hi']:.1f}%")
                
                st.markdown("---")
                
                # Gezin & Demografie
                st.subheader("👨‍👩‍👧 Gezin & Demografie")
                col1, col2 = st.columns(2)
                with col1:
                    if 'pct_children' in location_data and pd.notna(location_data['pct_children']):
                        st.metric("Kinderen (0-14 jaar)", f"{location_data['pct_children']:.1f}%")
                    if 'pct_families' in location_data and pd.notna(location_data['pct_families']):
                        st.metric("Gezinnen met kinderen", f"{location_data['pct_families']:.1f}%")
                
                with col2:
                    if 'g_hhgro' in location_data and pd.notna(location_data['g_hhgro']):
                        st.metric("Gemiddelde huishoudgrootte", f"{location_data['g_hhgro']:.1f} personen")
                    if 'g_afs_sc' in location_data and pd.notna(location_data['g_afs_sc']):
                        st.metric("Afstand basisschool", f"{location_data['g_afs_sc']:.1f} km")
                
                st.markdown("---")
                
                # Herkomst
                st.subheader("🌍 Herkomst Bevolking")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'p_herk_nl' in location_data and pd.notna(location_data['p_herk_nl']):
                        st.metric("Nederlandse herkomst", f"{location_data['p_herk_nl']:.1f}%")
                with col2:
                    if 'p_herk_eur' in location_data and pd.notna(location_data['p_herk_eur']):
                        st.metric("Europese herkomst (excl. NL)", f"{location_data['p_herk_eur']:.1f}%")
                with col3:
                    if 'p_herk_neu' in location_data and pd.notna(location_data['p_herk_neu']):
                        st.metric("Buiten-Europese herkomst", f"{location_data['p_herk_neu']:.1f}%")
                
                st.markdown("---")
                
                # Ruimte & Omgeving
                st.subheader("🌳 Ruimte & Omgeving")
                col1, col2 = st.columns(2)
                with col1:
                    if 'area_per_person_m2' in location_data and pd.notna(location_data['area_per_person_m2']):
                        st.metric("Oppervlakte per persoon", f"{location_data['area_per_person_m2']:.0f} m²")
                    if 'pct_water' in location_data and pd.notna(location_data['pct_water']):
                        st.metric("Wateroppervlak", f"{location_data['pct_water']:.1f}%")
                    if 'groen_percentage' in location_data and pd.notna(location_data['groen_percentage']):
                        st.metric("Groenpercentage 🌳", f"{location_data['groen_percentage']:.1f}%")
                
                with col2:
                    if 'bev_dich' in location_data and pd.notna(location_data['bev_dich']):
                        st.metric("Bevolkingsdichtheid", f"{location_data['bev_dich']:.0f} per km²")
                    if 'a_opp_ha' in location_data and pd.notna(location_data['a_opp_ha']):
                        st.metric("Totale oppervlakte", f"{location_data['a_opp_ha']:.0f} ha")
                    if 'nap_hoogte_gem' in location_data and pd.notna(location_data['nap_hoogte_gem']):
                        st.metric("NAP Hoogte 🏔️", f"{location_data['nap_hoogte_gem']:.1f} m")
                
                st.markdown("---")
                
                # Voorzieningen
                st.subheader("🏥 Voorzieningen")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'g_afs_hp' in location_data and pd.notna(location_data['g_afs_hp']):
                        st.metric("Afstand huisarts", f"{location_data['g_afs_hp']:.1f} km")
                with col2:
                    if 'g_afs_gs' in location_data and pd.notna(location_data['g_afs_gs']):
                        st.metric("Afstand supermarkt", f"{location_data['g_afs_gs']:.1f} km")
                with col3:
                    if 'g_afs_sc' in location_data and pd.notna(location_data['g_afs_sc']):
                        st.metric("Afstand school", f"{location_data['g_afs_sc']:.1f} km")
                
                # Transport & Bereikbaarheid
                st.subheader("🚆 Transport & Bereikbaarheid")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'g_afs_trein' in location_data and pd.notna(location_data['g_afs_trein']):
                        st.metric("Treinstation", f"{location_data['g_afs_trein']:.1f} km")
                with col2:
                    if 'g_afs_overstap' in location_data and pd.notna(location_data['g_afs_overstap']):
                        st.metric("Overstapstation", f"{location_data['g_afs_overstap']:.1f} km")
                with col3:
                    if 'g_afs_oprit' in location_data and pd.notna(location_data['g_afs_oprit']):
                        st.metric("Snelweg-oprit", f"{location_data['g_afs_oprit']:.1f} km")
                with col4:
                    if 'g_afs_bieb' in location_data and pd.notna(location_data['g_afs_bieb']):
                        st.metric("Bibliotheek", f"{location_data['g_afs_bieb']:.1f} km")
                
                # NEW: Trend Information Section
                if 'trend_score' in location_data.index and pd.notna(location_data['trend_score']):
                    st.markdown("---")
                    st.subheader("📈 Trends & Dynamiek")
                    
                    trend_score = location_data['trend_score']
                    
                    # Classify trend
                    if trend_score > 10:
                        trend_label = "🟢 Sterk groeiend"
                    elif trend_score > 5:
                        trend_label = "🟢 Groeiend"
                    elif trend_score > -5:
                        trend_label = "🟡 Stabiel"
                    elif trend_score > -10:
                        trend_label = "🔴 Krimpend"
                    else:
                        trend_label = "🔴 Sterk krimpend"
                    
                    st.metric("Trend Score", f"{trend_score:+.1f}", 
                             help="Positief = groei, Negatief = krimp")
                    st.markdown(f"**Status:** {trend_label}")
                    
                    # Show key trend indicators
                    st.markdown("**Belangrijkste trends:**")
                    
                    trend_indicators = [
                        ('pct_children', 'Kinderen 0-14 jaar'),
                        ('bev_dich', 'Bevolking'),
                        ('ses_overall', 'SES Overall'),
                    ]
                    
                    for ind_key, ind_name in trend_indicators:
                        trend_pct_col = f'trend_{ind_key}_pct'
                        trend_dir_col = f'trend_{ind_key}_dir'
                        
                        if trend_pct_col in location_data.index and pd.notna(location_data[trend_pct_col]):
                            pct = location_data[trend_pct_col]
                            direction = location_data.get(trend_dir_col, 'stabiel')
                            
                            if direction == 'stijgend':
                                arrow = '↑'
                                color_emoji = '🟢'
                            elif direction == 'dalend':
                                arrow = '↓'
                                color_emoji = '🔴'
                            else:
                                arrow = '↔'
                                color_emoji = '🟡'
                            
                            st.markdown(f"{color_emoji} {ind_name}: {arrow} **{pct:+.1f}%**")
                    
                    st.caption("📊 Trends berekend uit CBS KWB data 2020-2025")
                
                # NEW: Veiligheid Section
                if 'crime_rate' in location_data.index and pd.notna(location_data['crime_rate']):
                    st.markdown("---")
                    st.subheader("🚨 Veiligheid")
                    
                    crime_rate = location_data['crime_rate']
                    
                    # Context interpretation
                    if crime_rate < 20:
                        context = "🟢 Laag (top 25%)"
                    elif crime_rate < 40:
                        context = "🟡 Gemiddeld-laag"
                    elif crime_rate < 60:
                        context = "🟡 Gemiddeld-hoog"
                    else:
                        context = "🔴 Hoog (bottom 25%)"
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Geregistreerde Misdrijven", 
                            f"{crime_rate:.1f}",
                            help="Aantal geregistreerde misdrijven per 1000 inwoners per jaar (Politie Open Data)"
                        )
                        st.caption(f"per 1000 inwoners")
                    
                    with col2:
                        st.markdown(f"**Classificatie:**")
                        st.markdown(f"### {context}")
                    
                    # Benchmarks
                    st.caption("📊 Benchmark: Nationaal mediaan ~29/1000, grote steden 60-90/1000, landelijk 10-20/1000")
                    
                    # Show trend if available
                    if 'trend_crime_rate_pct' in location_data.index and pd.notna(location_data['trend_crime_rate_pct']):
                        trend_pct = location_data['trend_crime_rate_pct']
                        if abs(trend_pct) >= 6:  # Significant change over 3 years
                            trend_emoji = "🟢" if trend_pct < 0 else "🔴"
                            trend_text = "dalend" if trend_pct < 0 else "stijgend"
                            st.info(f"{trend_emoji} Trend (2020→2025): {abs(trend_pct):.1f}% {trend_text}")
                    
                    # Show inbraak separately if available
                    if 'inbraak_rate' in location_data.index and pd.notna(location_data['inbraak_rate']):
                        st.markdown("---")
                        st.markdown("**🏠 Inbraak Specifiek:**")
                        
                        inbraak_rate = location_data['inbraak_rate']
                        
                        # Inbraak context
                        if inbraak_rate < 3:
                            inbraak_context = "🟢 Zeer laag"
                        elif inbraak_rate < 6:
                            inbraak_context = "🟡 Laag"
                        elif inbraak_rate < 10:
                            inbraak_context = "🟠 Gemiddeld"
                        else:
                            inbraak_context = "🔴 Hoog"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Inbraak Rate", f"{inbraak_rate:.1f}", help="Inbraken per 1000 inwoners per jaar")
                            st.caption("per 1000 inwoners")
                        with col2:
                            st.markdown(f"**Niveau:**")
                            st.markdown(f"### {inbraak_context}")
                        
                        # Show trend if available
                        if 'trend_inbraak_rate_pct' in location_data.index and pd.notna(location_data['trend_inbraak_rate_pct']):
                            trend_pct = location_data['trend_inbraak_rate_pct']
                            if abs(trend_pct) >= 6:
                                trend_emoji = "🟢" if trend_pct < 0 else "🔴"
                                trend_text = "dalend" if trend_pct < 0 else "stijgend"
                                st.info(f"{trend_emoji} Inbraak trend (2020→2025): {abs(trend_pct):.1f}% {trend_text}")
                        
                        st.caption("💡 Benchmark: Landelijk mediaan ~5/1000, grote steden 8-12/1000")
                    
                    st.caption("📊 Bron: Politie Open Data (data.politie.nl)")
    
    # TAB 4: Vergelijking - Ruwe CBS Data
    with tab4:
        st.header("📈 Vergelijk Locaties - CBS Kerncijfers")
        
        # Enable cross-level comparison
        st.markdown("💡 **Tip:** Je kunt nu ook verschillende niveaus vergelijken (gemeente vs wijk vs buurt)")
        
        # Option to compare across levels
        compare_mode = st.radio(
            "Vergelijkingsmodus:",
            options=['Binnen huidig niveau', 'Alle niveaus (cross-level)'],
            horizontal=True
        )
        
        if compare_mode == 'Binnen huidig niveau':
            # Original: only within current geo_level
            location_options = sorted(df_filtered['display_name'].dropna().unique())
        else:
            # NEW: Allow comparison across all levels with level labels
            # Create labeled options: "Amsterdam (Gemeente)", "Centrum (Wijk, Amsterdam)", etc.
            df_with_labels = df.copy()
            level_labels = {
                'land': 'Land',
                'provincie': 'Provincie', 
                'gemeente': 'Gemeente',
                'wijk': 'Wijk',
                'buurt': 'Buurt'
            }
            
            def create_label(row):
                """Create label with gemeente for wijk/buurt to avoid duplicates"""
                if pd.isna(row['display_name']):
                    return None
                
                level = level_labels.get(row['geo_level'], row['geo_level'])
                name = row['display_name']
                
                # For wijk and buurt: add gemeente name to disambiguate
                if row['geo_level'] in ['wijk', 'buurt']:
                    gemeente = row.get('gm_naam', '')
                    if pd.notna(gemeente):
                        return f"{name} ({level} in {gemeente})"
                    else:
                        return f"{name} ({level})"
                else:
                    # For land, provincie, gemeente: just add level
                    return f"{name} ({level})"
            
            df_with_labels['display_with_level'] = df_with_labels.apply(create_label, axis=1)
            
            # Create mapping from labeled name to gwb_code_10 for unique identification
            label_to_code = dict(zip(df_with_labels['display_with_level'], df_with_labels['gwb_code_10']))
            location_options = sorted(df_with_labels['display_with_level'].dropna().unique())
        
        if len(location_options) == 0:
            st.warning("Geen locaties beschikbaar. Pas filters aan.")
        else:
            selected_locations = st.multiselect(
                "Selecteer locaties om te vergelijken (2-5):",
                options=location_options,
                max_selections=5,
                key='comparison_locations'
            )
            
            # Update URL to remember this tab when rerun happens
            if 'comparison_locations' in st.session_state and st.session_state.get('comparison_locations'):
                st.query_params['tab'] = '3'  # Tab 4 is index 3
            
            if len(selected_locations) < 2:
                st.info("Selecteer minimaal 2 locaties om te vergelijken")
            else:
                # Get comparison data
                if compare_mode == 'Binnen huidig niveau':
                    comparison_df = df_filtered[df_filtered['display_name'].isin(selected_locations)]
                else:
                    # Use gwb_code_10 for unique identification
                    # Map selected labeled names to their unique codes
                    selected_codes = [label_to_code.get(loc) for loc in selected_locations if loc in label_to_code]
                    
                    # Filter dataframe by unique codes
                    comparison_df = df[df['gwb_code_10'].isin(selected_codes)].copy()
                    
                    level_labels = {
                        'land': 'Land',
                        'provincie': 'Provincie', 
                        'gemeente': 'Gemeente',
                        'wijk': 'Wijk',
                        'buurt': 'Buurt'
                    }
                    
                    def create_label_for_comparison(row):
                        """Create consistent label for comparison table"""
                        if pd.isna(row['display_name']):
                            return None
                        
                        level = level_labels.get(row['geo_level'], row['geo_level'])
                        name = row['display_name']
                        
                        # For wijk and buurt: add gemeente name
                        if row['geo_level'] in ['wijk', 'buurt']:
                            gemeente = row.get('gm_naam', '')
                            if pd.notna(gemeente):
                                return f"{name} ({level} in {gemeente})"
                            else:
                                return f"{name} ({level})"
                        else:
                            return f"{name} ({level})"
                    
                    # Add labeled names for display
                    comparison_df['display_with_level'] = comparison_df.apply(create_label_for_comparison, axis=1)
                
                # Select key indicators for comparison
                indicators = []
                
                # Add custom score if configured
                if 'custom_indicator_weights' in st.session_state and st.session_state['custom_indicator_weights']:
                    indicators.append(('🎯 Mijn Custom Score', 'custom_score', 'score'))
                
                # Standard indicators - COMPLETE list (29 indicators)
                indicators.extend([
                    # SES (4)
                    ('SES Overall', 'ses_overall', 'percentiel'),
                    ('SES Welvaart', 'ses_welvaart', 'percentiel'),
                    ('SES Inkomen', 'ses_inkomen', 'percentiel'),
                    ('SES Onderwijs', 'ses_onderwijs', 'percentiel'),
                    # Welvaart & Woning (6)
                    ('Inkomen', 'g_ink_pi', '€'),
                    ('WOZ waarde', 'g_wozbag', '€'),
                    ('Participatie', 'p_arb_pp', '%'),
                    ('Laag inkomen', 'p_ink_li', '%'),
                    ('Hoog inkomen', 'p_ink_hi', '%'),
                    ('Koopwoningen', 'p_koopw', '%'),
                    # Gezin (3)
                    ('Kinderen %', 'pct_children', '%'),
                    ('Gezinnen %', 'pct_families', '%'),
                    ('Huishoudgrootte', 'g_hhgro', 'pers'),
                    # Herkomst (3)
                    ('Herkomst: Nederland', 'p_herk_nl', '%'),
                    ('Herkomst: Europa (excl. NL)', 'p_herk_eur', '%'),
                    ('Herkomst: Buiten-Europa', 'p_herk_neu', '%'),
                    # Veiligheid (2)
                    ('🚨 Misdrijven (totaal)', 'crime_rate', 'per 1000'),
                    ('🏠 Inbraak', 'inbraak_rate', 'per 1000'),
                    # Voorzieningen (4)
                    ('Afs. basisschool', 'g_afs_sc', 'km'),
                    ('Afs. kinderopvang', 'g_afs_kv', 'km'),
                    ('Afs. huisarts', 'g_afs_hp', 'km'),
                    ('Afs. supermarkt', 'g_afs_gs', 'km'),
                    # Transport & Bereikbaarheid (4)
                    ('Afs. treinstation 🚆', 'g_afs_trein', 'km'),
                    ('Afs. overstapstation 🚉', 'g_afs_overstap', 'km'),
                    ('Afs. snelweg-oprit 🚗', 'g_afs_oprit', 'km'),
                    ('Afs. bibliotheek 📚', 'g_afs_bieb', 'km'),
                    # Ruimte & Groen (6)
                    ('Oppervlakte/pers', 'area_per_person_m2', 'm²'),
                    ('Water %', 'pct_water', '%'),
                    ('Dichtheid', 'bev_dich', '/km²'),
                    ('Totale opp.', 'a_opp_ha', 'ha'),
                    ('Groen % 🌳', 'groen_percentage', '%'),
                    ('NAP Hoogte 🏔️', 'nap_hoogte_gem', 'm'),
                ])
                
                # Create comparison table
                comparison_data = []
                
                # Determine which column to use for location names
                if compare_mode == 'Alle niveaus (cross-level)' and 'display_with_level' in comparison_df.columns:
                    location_col = 'display_with_level'
                else:
                    location_col = 'display_name'
                
                for indicator_name, col, unit in indicators:
                    if col in comparison_df.columns:
                        row_data = {'Indicator': f"{indicator_name} ({unit})"}
                        for idx, row in comparison_df.iterrows():
                            loc_name = row[location_col]
                            value = row[col]
                            if pd.notna(value):
                                if unit == 'score':
                                    row_data[loc_name] = f"{value:.1f}/100"
                                elif unit == '€':
                                    # CBS data is in thousands (x1000)
                                    row_data[loc_name] = f"€{value*1000:,.0f}"
                                elif unit == '%':
                                    row_data[loc_name] = f"{value:.1f}%"
                                elif unit == 'per 1000':
                                    row_data[loc_name] = f"{value:.1f}"
                                elif unit == 'km':
                                    row_data[loc_name] = f"{value:.1f} km"
                                elif unit == 'm²':
                                    row_data[loc_name] = f"{value:.0f} m²"
                                elif unit == '/km²':
                                    row_data[loc_name] = f"{value:.0f}/km²"
                                else:
                                    row_data[loc_name] = f"{value:.1f}"
                            else:
                                row_data[loc_name] = "N/A"
                        comparison_data.append(row_data)
                
                comparison_table = pd.DataFrame(comparison_data)
                st.dataframe(comparison_table, width='stretch', hide_index=True)
                
                # Download button
                csv = comparison_table.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download vergelijking als CSV",
                    data=csv,
                    file_name=f"vergelijking_{len(selected_locations)}_locaties.csv",
                    mime="text/csv"
                )


if __name__ == '__main__':
    main()
