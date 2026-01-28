"""
RIVM Groenpercentage WFS Loader

Load green percentage data from RIVM Atlas Natuurlijk Kapitaal.
Data source: RIVM 2022 Groenpercentage per gemeente/wijk/buurt
WFS: https://data.rivm.nl/geo/ank/wfs
"""

import geopandas as gpd
import pandas as pd
from typing import Optional
import requests


class RIVMGroenLoader:
    """Load RIVM green percentage data via WFS"""
    
    WFS_URL = "https://data.rivm.nl/geo/ank/wfs"
    
    # Layer names for different geographic levels
    LAYERS = {
        'gemeente': 'ank:rivm_2022_groenpercentage_kaart_per_gemeente',
        'wijk': 'ank:rivm_2022_groenpercentage_kaart_per_wijk',
        'buurt': 'ank:rivm_2022_groenpercentage_kaart_per_buurt',
    }
    
    def __init__(self):
        """Initialize loader"""
        pass
    
    def _fetch_wfs_layer(self, layer_name: str) -> gpd.GeoDataFrame:
        """
        Fetch a layer from RIVM WFS
        
        Args:
            layer_name: WFS layer name (e.g., 'ank:rivm_2022_groenpercentage_kaart_per_gemeente')
            
        Returns:
            GeoDataFrame with green percentage data
        """
        print(f"📡 Fetching {layer_name} from RIVM WFS...")
        
        params = {
            'service': 'WFS',
            'version': '2.0.0',
            'request': 'GetFeature',
            'typename': layer_name,
            'outputFormat': 'application/json'
        }
        
        try:
            response = requests.get(self.WFS_URL, params=params, timeout=60)
            response.raise_for_status()
            
            # Load as GeoDataFrame
            gdf = gpd.read_file(response.content)
            print(f"   ✅ Loaded {len(gdf)} records")
            
            return gdf
            
        except Exception as e:
            print(f"   ❌ Error loading layer: {e}")
            raise
    
    def load_groen_data(self, levels: list = ['gemeente', 'wijk', 'buurt']) -> pd.DataFrame:
        """
        Load green percentage data for specified geographic levels
        
        Args:
            levels: List of geographic levels ('gemeente', 'wijk', 'buurt')
            
        Returns:
            DataFrame with columns: gwb_code_10, groen_percentage
        """
        print("\n🌳 LOADING RIVM GROENPERCENTAGE DATA")
        print("=" * 70)
        
        all_data = []
        
        for level in levels:
            if level not in self.LAYERS:
                print(f"⚠️  Unknown level: {level}, skipping...")
                continue
            
            layer_name = self.LAYERS[level]
            
            try:
                gdf = self._fetch_wfs_layer(layer_name)
                
                # Extract relevant columns
                # CBS code: level-specific code (wk_code for wijk, bu_code for buurt, gm_code for gemeente)
                code_col = None
                if level == 'gemeente':
                    possible_codes = ['gm_code', 'gwb_code_10', 'statcode', 'code']
                elif level == 'wijk':
                    possible_codes = ['wk_code', 'gwb_code_10', 'statcode', 'code']
                elif level == 'buurt':
                    possible_codes = ['bu_code', 'gwb_code_10', 'statcode', 'code']
                else:
                    possible_codes = ['gwb_code_10', 'statcode', 'code']
                
                for possible_name in possible_codes:
                    if possible_name in gdf.columns:
                        code_col = possible_name
                        break
                
                # Green percentage: _mean column contains the percentage
                groen_col = None
                for possible_name in ['_mean', 'groenpercentage', 'groen_percentage', 'mean', 'percentage']:
                    if possible_name in gdf.columns:
                        groen_col = possible_name
                        break
                
                if code_col and groen_col:
                    # Extract data (drop geometry for efficiency)
                    # RIVM codes are already in correct format (GM0363, WK036301, BU03630100)
                    level_data = []
                    for idx, row in gdf.iterrows():
                        code = str(row[code_col])
                        groen = row[groen_col]
                        level_data.append({'gwb_code_10': code, 'groen_percentage': groen})
                    
                    df_level = pd.DataFrame(level_data)
                    
                    all_data.append(df_level)
                    print(f"   ✅ {level:10s}: {len(df_level):,} records added (sample: {df_level['gwb_code_10'].iloc[0]})")
                    
                else:
                    print(f"   ⚠️  {level:10s}: Could not find required columns")
                    print(f"      code_col={code_col}, groen_col={groen_col}")
                    print(f"      Available columns: {list(gdf.columns)}")
                    
            except Exception as e:
                print(f"   ❌ {level:10s}: Error - {e}")
        
        if not all_data:
            raise ValueError("No data loaded from any level!")
        
        # Combine all levels - NO duplicate removal! 
        # Each level has unique codes (GM, WK, BU are different)
        df_combined = pd.concat(all_data, ignore_index=True)
        
        # Convert percentage to float
        df_combined['groen_percentage'] = pd.to_numeric(df_combined['groen_percentage'], errors='coerce')
        
        print(f"\n✅ Total records: {len(df_combined):,}")
        print(f"   Gemeente records: {len([c for c in df_combined['gwb_code_10'] if c.startswith('GM')])}")
        print(f"   Wijk records: {len([c for c in df_combined['gwb_code_10'] if c.startswith('WK')])}")
        print(f"   Buurt records: {len([c for c in df_combined['gwb_code_10'] if c.startswith('BU')])}")
        print(f"   Coverage: {(df_combined['groen_percentage'].notna().sum() / len(df_combined) * 100):.1f}%")
        print(f"   Range: {df_combined['groen_percentage'].min():.1f}% - {df_combined['groen_percentage'].max():.1f}%")
        print(f"   Mean: {df_combined['groen_percentage'].mean():.1f}%")
        
        print("\n" + "=" * 70)
        
        return df_combined
    
    def load_for_dashboard(self) -> pd.DataFrame:
        """
        Convenience method to load all levels for dashboard
        
        Returns:
            DataFrame ready to merge with main_data
        """
        return self.load_groen_data(levels=['gemeente', 'wijk', 'buurt'])


def load_rivm_groen_for_dashboard() -> pd.DataFrame:
    """
    Convenience function to load RIVM groen data
    
    Returns:
        DataFrame with gwb_code_10 and groen_percentage
    """
    loader = RIVMGroenLoader()
    return loader.load_for_dashboard()


if __name__ == '__main__':
    # Test the loader
    print("🧪 TESTING RIVM GROEN LOADER\n")
    
    loader = RIVMGroenLoader()
    df = loader.load_groen_data(levels=['gemeente'])  # Test with just gemeente first
    
    print("\n\n📊 RESULT SAMPLE:")
    print(df.head(10))
    print(f"\n✅ Test complete! Loaded {len(df):,} records")
