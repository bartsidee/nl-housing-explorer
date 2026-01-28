"""
PDOK Geo Data Loader
Laadt gemeente, wijk en buurt grenzen via PDOK WFS (geen API key nodig!)
"""
import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
import json
from io import BytesIO


class PDOKGeoLoader:
    """Load geographic boundaries from PDOK WFS service"""
    
    # PDOK WFS endpoints (publiekelijk toegankelijk, geen API key!)
    # Updated to 2024 for consistency with CBS data 2024/2025
    WFS_BASE_URL = "https://service.pdok.nl/cbs/gebiedsindelingen/2024/wfs/v1_0"
    
    # Layer names (gevonden via GetCapabilities)
    LAYER_GEMEENTE = "gebiedsindelingen:gemeente_gegeneraliseerd"
    LAYER_WIJK = "gebiedsindelingen:wijk_gegeneraliseerd"
    LAYER_BUURT = "gebiedsindelingen:buurt_gegeneraliseerd"
    
    def __init__(self, cache_dir: str = 'data/geo/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def _fetch_with_pagination(self, params: dict, layer_name: str, timeout: int = 180) -> gpd.GeoDataFrame:
        """
        Fetch WFS data with pagination to handle large datasets (>1000 features)
        PDOK has a default limit of ~1000 features per request
        """
        all_features = []
        start_index = 0
        page_size = 1000  # PDOK default
        
        while True:
            # Add pagination parameters
            paginated_params = params.copy()
            paginated_params['startIndex'] = start_index
            paginated_params['count'] = page_size
            
            print(f"  Fetching {layer_name}: offset {start_index}...")
            
            response = requests.get(self.WFS_BASE_URL, params=paginated_params, timeout=timeout)
            response.raise_for_status()
            
            # Parse GeoJSON
            gdf_page = gpd.read_file(BytesIO(response.content))
            
            if len(gdf_page) == 0:
                break  # No more features
            
            all_features.append(gdf_page)
            print(f"    Got {len(gdf_page)} features (total so far: {sum(len(gdf) for gdf in all_features)})")
            
            # Check if we got less than page_size (last page)
            if len(gdf_page) < page_size:
                break
            
            start_index += page_size
        
        # Combine all pages
        if not all_features:
            # Return empty GeoDataFrame with correct structure
            return gpd.GeoDataFrame()
        
        gdf_combined = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True))
        print(f"  ✅ Total downloaded: {len(gdf_combined)} features")
        
        return gdf_combined
    
    def get_gemeenten(self, use_cache: bool = True) -> gpd.GeoDataFrame:
        """
        Haal gemeente grenzen op via PDOK WFS
        
        Returns:
            GeoDataFrame met gemeente polygonen
        """
        cache_file = self.cache_dir / 'gemeenten_2024.geojson'
        
        if use_cache and cache_file.exists():
            print(f"Loading gemeenten from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        print("Downloading gemeenten from PDOK WFS...")
        
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetFeature',
            'typeName': self.LAYER_GEMEENTE,
            'outputFormat': 'json',
            'srsName': 'EPSG:4326'  # WGS84 voor Leaflet/Folium
        }
        
        gdf = self._fetch_with_pagination(params, 'gemeenten', timeout=60)
        
        print(f"Downloaded {len(gdf)} gemeenten")
        
        # Save to cache
        gdf.to_file(cache_file, driver='GeoJSON')
        print(f"Cached to: {cache_file}")
        
        return gdf
    
    def get_wijken(self, gemeente_code: str = None, use_cache: bool = True) -> gpd.GeoDataFrame:
        """
        Haal wijk grenzen op via PDOK WFS
        
        Args:
            gemeente_code: Optioneel filter op gemeente (bijv. 'GM0363' voor Amsterdam)
            
        Returns:
            GeoDataFrame met wijk polygonen
        """
        cache_file = self.cache_dir / f'wijken_2024_{gemeente_code or "all"}.geojson'
        
        if use_cache and cache_file.exists():
            print(f"Loading wijken from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        print(f"Downloading wijken from PDOK WFS{f' for {gemeente_code}' if gemeente_code else ''}...")
        
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetFeature',
            'typeName': self.LAYER_WIJK,
            'outputFormat': 'json',
            'srsName': 'EPSG:4326'
        }
        
        # Add filter if gemeente specified
        if gemeente_code:
            # CQL filter voor gemeente
            params['CQL_FILTER'] = f"gemeentecode='{gemeente_code}'"
        
        gdf = self._fetch_with_pagination(params, 'wijken', timeout=120)
        print(f"Downloaded {len(gdf)} wijken")
        
        # Save to cache
        gdf.to_file(cache_file, driver='GeoJSON')
        print(f"Cached to: {cache_file}")
        
        return gdf
    
    def get_buurten(self, gemeente_code: str = None, use_cache: bool = True) -> gpd.GeoDataFrame:
        """
        Haal buurt grenzen op via PDOK WFS
        
        Args:
            gemeente_code: Optioneel filter op gemeente
            
        Returns:
            GeoDataFrame met buurt polygonen
        """
        cache_file = self.cache_dir / f'buurten_2024_{gemeente_code or "all"}.geojson'
        
        if use_cache and cache_file.exists():
            print(f"Loading buurten from cache: {cache_file}")
            return gpd.read_file(cache_file)
        
        print(f"Downloading buurten from PDOK WFS{f' for {gemeente_code}' if gemeente_code else ''}...")
        
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetFeature',
            'typeName': self.LAYER_BUURT,
            'outputFormat': 'json',
            'srsName': 'EPSG:4326'
        }
        
        if gemeente_code:
            params['CQL_FILTER'] = f"gemeentecode='{gemeente_code}'"
        
        gdf = self._fetch_with_pagination(params, 'buurten', timeout=240)
        print(f"Downloaded {len(gdf)} buurten")
        
        gdf.to_file(cache_file, driver='GeoJSON')
        print(f"Cached to: {cache_file}")
        
        return gdf
    
    def merge_with_scores(self, gdf: gpd.GeoDataFrame, scores_df, 
                          geo_key: str = 'statcode') -> gpd.GeoDataFrame:
        """
        Merge geographic data met score data
        
        Args:
            gdf: GeoDataFrame met grenzen
            scores_df: DataFrame met scores
            geo_key: Kolom naam voor CBS code in gdf
            
        Returns:
            Merged GeoDataFrame
        """
        # Clean codes for matching
        if geo_key in gdf.columns:
            gdf['match_code'] = gdf[geo_key].str.replace('GM', '').str.replace('WK', '').str.replace('BU', '')
        
        # Merge
        merged = gdf.merge(
            scores_df,
            left_on='match_code',
            right_on='gwb_code_10',
            how='left'
        )
        
        print(f"Merged: {len(merged)} features, {merged['overall_score'].notna().sum()} with scores")
        
        return merged


def main():
    """Test PDOK geo loader"""
    loader = PDOKGeoLoader()
    
    print("\n" + "=" * 60)
    print("TESTING PDOK GEO LOADER (NO API KEY NEEDED!)")
    print("=" * 60)
    
    # Test 1: Download gemeenten
    print("\n1. Downloading ALL gemeenten...")
    gemeenten = loader.get_gemeenten(use_cache=False)
    print(f"✅ Success! {len(gemeenten)} gemeenten downloaded")
    print(f"Columns: {gemeenten.columns.tolist()}")
    print(f"\nSample:")
    print(gemeenten[['statnaam', 'statcode']].head())
    
    # Test 2: Download wijken voor Amsterdam
    print("\n2. Downloading wijken for Amsterdam (GM0363)...")
    wijken_adam = loader.get_wijken(gemeente_code='GM0363', use_cache=False)
    print(f"✅ Success! {len(wijken_adam)} wijken in Amsterdam")
    
    # Test 3: Show cache works
    print("\n3. Testing cache...")
    gemeenten_cached = loader.get_gemeenten(use_cache=True)
    print(f"✅ Cache works! Loaded from cache instantly")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print("\nPDOK WFS works perfect - geen API key nodig!")
    print("Ready to integrate in dashboard!")


if __name__ == '__main__':
    main()
