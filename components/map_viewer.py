"""
Map visualization component using Folium
"""
import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from pdok_geo_loader import PDOKGeoLoader


import math


# Color scheme mapping for different indicator types
INDICATOR_COLORSCHEMES = {
    # POSITIVE: Higher = Better (Blue scale)
    'positive': {
        'colorscheme': 'YlGnBu',
        'indicators': [
            'ses_overall', 'ses_welvaart', 'ses_inkomen', 'ses_onderwijs',
            'g_ink_pi', 'g_wozbag', 'p_arb_pp', 'p_ink_hi', 'p_koopw',
            'pct_children', 'pct_families', 'g_hhgro',
            'area_per_person_m2', 'pct_water',
            # NEW: Groen percentage (higher = greener = better)
            'groen_percentage',
            # NEW: Custom score (higher = better)
            'custom_score',
        ]
    },
    # NEGATIVE: Lower = Better (Green-Red scale reversed, green=low=good)
    'negative': {
        'colorscheme': 'RdYlGn_r',  # _r = reversed! Groen=laag, Rood=hoog
        'indicators': [
            'g_afs_hp', 'g_afs_gs', 'g_afs_sc', 'g_afs_kv',
            # Transport & Bereikbaarheid: lager = beter (dichtbij = goed)
            'g_afs_trein', 'g_afs_overstap', 'g_afs_oprit', 'g_afs_bieb',
            'p_ink_li', 'bev_dich',
            # Veiligheid: lager = beter (minder misdrijven/inbraak = beter)
            'crime_rate', 'inbraak_rate'
        ]
    },
    # NEUTRAL: No value judgment (Purple scale)
    'neutral': {
        'colorscheme': 'BuPu',
        'indicators': [
            'a_opp_ha'
        ]
    },
    # DIVERGING: Both high and low can be significant (terrain elevation)
    'diverging': {
        'colorscheme': 'RdYlBu',  # Red (high) - Yellow (medium) - Blue (low)
        'indicators': [
            'nap_hoogte_gem',  # NAP height: Blue=under zeeniveau, Red=heuvels
        ]
    }
}


def get_colorscheme_for_indicator(indicator: str) -> str:
    """
    Get appropriate color scheme for an indicator
    
    Args:
        indicator: Indicator column name
        
    Returns:
        Folium colorscheme name
    """
    for category, config in INDICATOR_COLORSCHEMES.items():
        if indicator in config['indicators']:
            return config['colorscheme']
    
    # Default to positive (blue scale) if not specified
    return 'YlGnBu'


def get_indicator_interpretation(indicator: str) -> str:
    """
    Get interpretation hint for indicator
    
    Args:
        indicator: Indicator column name
        
    Returns:
        Interpretation string
    """
    for category, config in INDICATOR_COLORSCHEMES.items():
        if indicator in config['indicators']:
            if category == 'positive':
                return 'Blauw = hoger = beter'
            elif category == 'negative':
                return 'Groen = lager = beter'
            elif category == 'diverging':
                return 'Blauw = laag/onder zeeniveau | Rood = hoog/heuvels'
            else:
                return 'Paars = neutraal'
    
    return 'Blauw = hoger'


def calculate_zoom_level(lat_diff: float, lon_diff: float) -> int:
    """
    Calculate appropriate zoom level based on geographic bounds
    
    Args:
        lat_diff: Latitude span (degrees)
        lon_diff: Longitude span (degrees)
        
    Returns:
        Zoom level (1-18)
    """
    # Max dimension determines zoom
    max_diff = max(lat_diff, lon_diff)
    
    # Rough zoom level calculation
    # Netherlands spans ~3° lat and ~5° lon
    if max_diff > 5:
        return 7  # Country level
    elif max_diff > 2:
        return 8  # Large province
    elif max_diff > 1:
        return 9  # Province
    elif max_diff > 0.5:
        return 10  # Large gemeente
    elif max_diff > 0.2:
        return 11  # Medium gemeente
    elif max_diff > 0.1:
        return 12  # Small gemeente
    elif max_diff > 0.05:
        return 13  # Wijk level
    else:
        return 14  # Buurt level


def calculate_bounds_center_zoom(gdf: gpd.GeoDataFrame):
    """
    Calculate center and zoom level from GeoDataFrame bounds
    
    Args:
        gdf: GeoDataFrame with geometries
        
    Returns:
        Tuple (center_lat, center_lon, zoom_level)
    """
    if len(gdf) == 0:
        # Default to Netherlands
        return (52.2, 5.3, 7)
    
    # Get bounds [minx, miny, maxx, maxy]
    bounds = gdf.total_bounds
    
    # Calculate center
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # Calculate zoom based on span
    lon_diff = bounds[2] - bounds[0]
    lat_diff = bounds[3] - bounds[1]
    zoom = calculate_zoom_level(lat_diff, lon_diff)
    
    return (center_lat, center_lon, zoom)


def create_choropleth_map(gdf_with_scores, score_column='overall_score', 
                          auto_zoom: bool = True,
                          center_lat: float = None, center_lon: float = None, zoom_start: int = None,
                          use_fixed_scale: bool = True, baseline_data=None):
    """
    Create choropleth map with score colors
    
    Args:
        gdf_with_scores: GeoDataFrame with geometries and scores
        score_column: Column to visualize
        auto_zoom: Automatically calculate center and zoom from data bounds
        center_lat, center_lon: Manual map center (ignored if auto_zoom=True)
        zoom_start: Manual zoom level (ignored if auto_zoom=True)
        use_fixed_scale: Use fixed national baseline for color scale (default: True)
        baseline_data: Optional full dataset for baseline calculation (if None, uses gdf_with_scores)
        
    Returns:
        Folium Map object
    """
    # Create a copy to avoid modifying original data
    gdf_with_scores = gdf_with_scores.copy()
    
    # Add rounded display column for tooltip (1 decimal)
    tooltip_column = f'{score_column}_display'
    gdf_with_scores[tooltip_column] = gdf_with_scores[score_column].round(1)
    
    # Calculate optimal center and zoom if auto_zoom enabled
    if auto_zoom:
        center_lat, center_lon, zoom_start = calculate_bounds_center_zoom(gdf_with_scores)
        print(f"Auto-zoom: center=({center_lat:.3f}, {center_lon:.3f}), zoom={zoom_start}")
    else:
        # Use defaults if not provided
        center_lat = center_lat or 52.2
        center_lon = center_lon or 5.3
        zoom_start = zoom_start or 7
    
    # Get appropriate color scheme for this indicator
    colorscheme = get_colorscheme_for_indicator(score_column)
    interpretation = get_indicator_interpretation(score_column)
    
    # Calculate data-driven thresholds (quantile-based)
    # FIXED SCALE MODE: Use baseline (national) data for consistent color scale across all areas
    # DYNAMIC SCALE MODE: Use only visible data for maximum color differentiation
    
    if use_fixed_scale and baseline_data is not None:
        # Use baseline data (full national dataset) for threshold calculation
        # This ensures consistent colors across different geographic filters
        if score_column in baseline_data.columns:
            valid_data = baseline_data[baseline_data[score_column].notna()][score_column]
            scale_mode = "FIXED (landelijk)"
        else:
            # Fallback to visible data if column not in baseline
            valid_data = gdf_with_scores[gdf_with_scores[score_column].notna()][score_column]
            scale_mode = "DYNAMIC (gefilterd)"
    else:
        # Use only visible (filtered) data
        valid_data = gdf_with_scores[gdf_with_scores[score_column].notna()][score_column]
        scale_mode = "DYNAMIC (gefilterd)"
    
    if len(valid_data) > 0:
        # Get the actual data that will be displayed on the map
        # This is critical to avoid bin errors when filtered data has different range
        map_data = gdf_with_scores[gdf_with_scores[score_column].notna()][score_column]
        
        if len(map_data) == 0:
            threshold_scale = None
        else:
            # Use the displayed data's range for thresholds
            map_min = map_data.min()
            map_max = map_data.max()
            data_range = map_max - map_min
            
            if data_range < 0.001:
                # No meaningful variation - no thresholds needed
                threshold_scale = None
            else:
                # Use quantile-based bins on the BASELINE data for consistent colors
                # But ensure they cover the MAP data range
                quantiles = [i/20 for i in range(21)]  # 0, 0.05, 0.10, ... 0.95, 1.0
                threshold_scale = [valid_data.quantile(q) for q in quantiles]
                
                # Remove duplicates and sort
                threshold_scale = sorted(list(set([round(x, 2) for x in threshold_scale])))
                
                # CRITICAL: Ensure the thresholds cover the actual MAP data range
                # Expand thresholds if needed to include all map values
                if threshold_scale[0] > map_min:
                    # Add a bin below the minimum to catch low outliers
                    threshold_scale.insert(0, round(map_min - 0.01, 2))
                if threshold_scale[-1] < map_max:
                    # Add a bin above the maximum to catch high outliers
                    threshold_scale.append(round(map_max + 0.01, 2))
                
                # Ensure we have enough unique thresholds
                if len(threshold_scale) < 3:
                    threshold_scale = None
    else:
        threshold_scale = None
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add choropleth layer with dynamic color scheme and data-driven thresholds
    legend_name = f"{score_column.replace('_', ' ').title()} ({interpretation})"
    
    choropleth_params = {
        'geo_data': gdf_with_scores,
        'data': gdf_with_scores,
        'columns': ['statcode', score_column],
        'key_on': 'feature.properties.statcode',
        'fill_color': colorscheme,
        'fill_opacity': 0.7,
        'line_opacity': 0.2,
        'legend_name': legend_name,
        'nan_fill_color': 'lightgray'
    }
    
    # Add custom thresholds if calculated
    if threshold_scale:
        choropleth_params['threshold_scale'] = threshold_scale
        print(f"Color scale: {scale_mode} - Thresholds: {[f'{x:.1f}' for x in threshold_scale]}")
    
    folium.Choropleth(**choropleth_params).add_to(m)
    
    # Add tooltips
    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    
    highlight_function = lambda x: {'fillColor': '#000000',
                                     'color':'#000000',
                                     'fillOpacity': 0.50,
                                     'weight': 0.1}
    
    tooltip = folium.features.GeoJsonTooltip(
        fields=['statnaam', tooltip_column],
        aliases=['Locatie:', f'{score_column}:'],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    
    popup = folium.features.GeoJsonPopup(
        fields=['statnaam', tooltip_column],
        aliases=['Locatie:', 'Score:'],
        localize=True,
        labels=True
    )
    
    folium.GeoJson(
        gdf_with_scores,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=tooltip,
        popup=popup
    ).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    return m


def load_and_merge_geo_data(df_scores, geo_level='gemeente', use_cache=True):
    """
    Load geographic boundaries and merge with scores
    
    Args:
        df_scores: DataFrame with scores
        geo_level: 'gemeente', 'wijk', or 'buurt'
        use_cache: Use cached geo data if available
        
    Returns:
        GeoDataFrame with geometries and scores
    """
    loader = PDOKGeoLoader()
    
    if geo_level == 'gemeente':
        gdf = loader.get_gemeenten(use_cache=use_cache)
    elif geo_level == 'wijk':
        gdf = loader.get_wijken(use_cache=use_cache)
    elif geo_level == 'buurt':
        gdf = loader.get_buurten(use_cache=use_cache)
    else:
        raise ValueError(f"Unknown geo_level: {geo_level}")
    
    # Merge with scores
    # PDOK uses 'statcode' which matches our 'gwb_code_10'
    gdf_merged = gdf.merge(
        df_scores,
        left_on='statcode',
        right_on='gwb_code_10',
        how='left'
    )
    
    return gdf_merged


def test_map():
    """Test map creation"""
    import pandas as pd
    
    # Load current processed data
    df = pd.read_csv('data/processed/current/main_data_with_trends.csv')
    
    # Filter gemeenten only
    df_gem = df[df['geo_level'] == 'gemeente']
    
    print(f"Loading geo data for {len(df_gem)} gemeenten...")
    gdf = load_and_merge_geo_data(df_gem, geo_level='gemeente', use_cache=True)
    
    print(f"Merged: {len(gdf)} gemeenten with geometries")
    print(f"Score coverage: {gdf['overall_score'].notna().sum()} / {len(gdf)}")
    
    # Create map
    print("Creating map...")
    m = create_choropleth_map(gdf, score_column='overall_score')
    
    # Save test map
    output_path = Path('data/geo/test_map.html')
    m.save(str(output_path))
    print(f"✅ Map saved to: {output_path}")
    print(f"   Open in browser to view!")


if __name__ == '__main__':
    test_map()
