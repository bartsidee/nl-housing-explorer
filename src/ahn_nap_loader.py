"""
AHN NAP Hoogte Loader

Loads and processes AHN (Actueel Hoogtebestand Nederland) elevation data from PDOK.
Calculates average NAP height per gemeente/wijk/buurt using WCS (Web Coverage Service).

Data source: AHN4 via PDOK WCS
API: https://service.pdok.nl/rws/ahn/wcs/v1_0

Author: AI Assistant
Date: 2026-01-27
"""

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional, Dict, Tuple
import time
from io import BytesIO
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class AHNNAPLoader:
    """Load and process AHN NAP elevation data via PDOK WCS"""
    
    # PDOK AHN4 WCS endpoint (Digital Surface Model - highest point including buildings/vegetation)
    WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
    
    # WCS coverage names
    # DSM = Digital Surface Model (hoogste punt, incl. gebouwen/vegetatie)
    # DTM = Digital Terrain Model (maaiveld, zonder gebouwen/vegetatie)
    COVERAGE_DSM = "dsm_05m"  # 0.5m resolution
    COVERAGE_DTM = "dtm_05m"  # 0.5m resolution - beter voor "waar je woont"
    
    # Coordinate systems
    CRS_WGS84 = "EPSG:4326"  # WGS84 (lat/lon) - used by PDOK geo data
    CRS_RD_NEW = "EPSG:28992"  # RD New - Dutch national grid, required by AHN WCS
    
    def __init__(self, cache_dir: str = 'data/geo/cache/ahn'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def _get_bbox_from_geometry(self, geometry, target_crs: str = CRS_RD_NEW) -> Tuple[float, float, float, float]:
        """
        Get bounding box from geometry, transformed to target CRS
        
        Args:
            geometry: Shapely geometry (assumed to be in WGS84/EPSG:4326)
            target_crs: Target coordinate system (default: RD New/EPSG:28992)
        
        Returns:
            (minx, miny, maxx, maxy) tuple in target CRS
        """
        # Convert geometry to GeoDataFrame for CRS transformation
        gdf_temp = gpd.GeoDataFrame([1], geometry=[geometry], crs=self.CRS_WGS84)
        
        # Transform to target CRS (RD New for AHN WCS)
        gdf_transformed = gdf_temp.to_crs(target_crs)
        
        # Get bounds in transformed coordinates
        bounds = gdf_transformed.geometry.iloc[0].bounds
        return bounds
    
    def _fetch_elevation_raster(
        self, 
        bbox: Tuple[float, float, float, float],
        coverage: str = "dtm_05m",
        width: int = 256,
        height: int = 256,
        timeout: int = 60
    ) -> Optional[np.ndarray]:
        """
        Fetch elevation raster from AHN WCS for given bounding box
        
        Args:
            bbox: Bounding box (minx, miny, maxx, maxy) in WGS84
            coverage: Coverage name (dsm_05m or dtm_05m)
            width: Requested image width in pixels
            height: Requested image height in pixels
            timeout: Request timeout in seconds
            
        Returns:
            2D numpy array with elevation values in meters NAP, or None if failed
        """
        minx, miny, maxx, maxy = bbox
        
        # WCS GetCoverage request
        # Note: PDOK AHN WCS uses RD New coordinates (EPSG:28992), not WGS84
        params = {
            'service': 'WCS',
            'version': '2.0.1',
            'request': 'GetCoverage',
            'coverageid': coverage,
            'format': 'image/tiff',  # PDOK supports 'image/tiff', not 'image/geotiff'
            # Subset must use RD New coordinates (bbox is already transformed)
            'subset': [
                f'x({minx},{maxx})',
                f'y({miny},{maxy})'
            ],
            'scalesize': f'x({width}),y({height})'
        }
        
        try:
            # Debug: show request details for first few requests
            # print(f"      WCS request: {self.WCS_URL}?{params}")
            
            response = requests.get(self.WCS_URL, params=params, timeout=timeout)
            response.raise_for_status()
            
            # Check if response is GeoTIFF
            content_type = response.headers.get('Content-Type', '')
            if 'geotiff' not in content_type.lower() and 'tiff' not in content_type.lower():
                # Sometimes WCS returns XML error messages
                if 'xml' in content_type.lower():
                    error_msg = response.text[:200] if len(response.text) < 200 else response.text[:200] + '...'
                    print(f"      ⚠️  WCS returned XML error: {error_msg}")
                else:
                    print(f"      ⚠️  Unexpected content type: {content_type}")
                return None
            
            # Read GeoTIFF using rasterio (if available) or PIL as fallback
            try:
                import rasterio
                from rasterio.io import MemoryFile
                
                with MemoryFile(response.content) as memfile:
                    with memfile.open() as dataset:
                        # Read first band (elevation)
                        elevation = dataset.read(1)
                        
                        # Handle nodata values
                        nodata = dataset.nodata
                        if nodata is not None:
                            elevation = np.where(elevation == nodata, np.nan, elevation)
                        
                        return elevation
                        
            except ImportError:
                # Fallback: PIL can read TIFF but won't preserve exact float values
                print("      ℹ️  rasterio not available, using PIL fallback (less accurate)")
                img = Image.open(BytesIO(response.content))
                elevation = np.array(img, dtype=np.float32)
                
                # Basic nodata handling (very high/low values are likely nodata)
                elevation = np.where(
                    (elevation < -1000) | (elevation > 10000), 
                    np.nan, 
                    elevation
                )
                
                return elevation
                
        except requests.exceptions.Timeout:
            print(f"      ⏱️  Timeout fetching elevation data")
            return None
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Request error: {e}")
            return None
        except Exception as e:
            print(f"      ❌ Error processing raster: {e}")
            return None
    
    def _calculate_mean_elevation_for_geometry(
        self,
        geometry,
        coverage: str = "dtm_05m",
        sample_size: int = 100
    ) -> Optional[float]:
        """
        Calculate mean elevation for a single geometry using centroid sampling
        
        Strategy: For large areas (gemeenten), using full bbox causes 400 errors.
        Instead, we sample a small area around the centroid for representative height.
        
        Args:
            geometry: Shapely geometry (in WGS84/EPSG:4326)
            coverage: AHN coverage to use
            sample_size: Size in meters for sample box around centroid
            
        Returns:
            Mean elevation in meters NAP, or None if failed
        """
        # Get centroid of geometry for representative sampling
        centroid_wgs84 = geometry.centroid
        
        # Transform centroid to RD New
        gdf_temp = gpd.GeoDataFrame([1], geometry=[centroid_wgs84], crs=self.CRS_WGS84)
        gdf_rd = gdf_temp.to_crs(self.CRS_RD_NEW)
        centroid_rd = gdf_rd.geometry.iloc[0]
        
        # Create small bounding box around centroid (sample_size x sample_size meters)
        buffer = sample_size / 2
        minx = centroid_rd.x - buffer
        maxx = centroid_rd.x + buffer
        miny = centroid_rd.y - buffer
        maxy = centroid_rd.y + buffer
        
        bbox = (minx, miny, maxx, maxy)
        
        # Fetch small raster (100m x 100m = 200 pixels at 0.5m resolution)
        # Use modest resolution to keep request small
        pixel_res = int(sample_size / 0.5)  # 0.5m AHN resolution
        width = min(pixel_res, 200)  # Cap at 200 pixels
        height = min(pixel_res, 200)
        
        # Fetch raster
        elevation_raster = self._fetch_elevation_raster(
            bbox=bbox,
            coverage=coverage,
            width=width,
            height=height
        )
        
        if elevation_raster is None:
            return None
        
        # Calculate mean (ignoring NaN values)
        mean_elevation = np.nanmean(elevation_raster)
        
        if np.isnan(mean_elevation):
            return None
        
        return float(mean_elevation)
    
    def calculate_nap_heights_for_regions(
        self,
        gdf: gpd.GeoDataFrame,
        code_column: str = 'statcode',
        name_column: str = 'statnaam',
        coverage: str = "dtm_05m",
        max_regions: Optional[int] = None,
        delay: float = 0.1
    ) -> pd.DataFrame:
        """
        Calculate average NAP height for multiple regions
        
        Args:
            gdf: GeoDataFrame with region geometries
            code_column: Column name containing region codes (e.g., 'statcode')
            name_column: Column name containing region names
            coverage: AHN coverage ('dtm_05m' or 'dsm_05m')
            max_regions: Maximum number of regions to process (for testing)
            delay: Delay between requests in seconds (to be nice to the API)
            
        Returns:
            DataFrame with columns: gwb_code_10, region_name, nap_hoogte_gem
        """
        print(f"\n🏔️  CALCULATING NAP HEIGHTS")
        print(f"   Coverage: {coverage}")
        print(f"   Regions: {len(gdf) if max_regions is None else max_regions}/{len(gdf)}")
        print("=" * 70)
        
        results = []
        
        # Process regions
        regions_to_process = gdf.head(max_regions) if max_regions else gdf
        total = len(regions_to_process)
        
        for idx, row in regions_to_process.iterrows():
            code = row[code_column]
            name = row.get(name_column, 'Unknown')
            geometry = row.geometry
            
            print(f"   [{idx+1}/{total}] {code} - {name}...", end=" ")
            
            # Calculate mean elevation
            mean_elevation = self._calculate_mean_elevation_for_geometry(
                geometry=geometry,
                coverage=coverage
            )
            
            if mean_elevation is not None:
                results.append({
                    'gwb_code_10': code,
                    'region_name': name,
                    'nap_hoogte_gem': round(mean_elevation, 2)
                })
                print(f"✅ {mean_elevation:.2f}m NAP")
            else:
                print(f"❌ Failed")
            
            # Be nice to the API
            if delay > 0 and idx < total - 1:
                time.sleep(delay)
        
        df_results = pd.DataFrame(results)
        
        print("\n" + "=" * 70)
        print(f"✅ Processed: {len(df_results)}/{total} regions")
        if len(df_results) > 0:
            print(f"   NAP range: {df_results['nap_hoogte_gem'].min():.2f}m - {df_results['nap_hoogte_gem'].max():.2f}m")
            print(f"   NAP mean: {df_results['nap_hoogte_gem'].mean():.2f}m")
        
        return df_results
    
    def load_or_calculate_gemeente_nap(
        self,
        use_cache: bool = True,
        force_recalculate: bool = False
    ) -> pd.DataFrame:
        """
        Load or calculate NAP heights for all gemeenten
        
        Args:
            use_cache: Try to load from cache first
            force_recalculate: Force recalculation even if cache exists
            
        Returns:
            DataFrame with gemeente NAP heights
        """
        cache_file = self.cache_dir / 'gemeente_nap_heights.csv'
        
        # Try cache first
        if use_cache and not force_recalculate and cache_file.exists():
            print(f"📂 Loading gemeente NAP heights from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            print(f"   ✅ Loaded {len(df)} gemeenten")
            return df
        
        # Load geometries
        from src.pdok_geo_loader import PDOKGeoLoader
        geo_loader = PDOKGeoLoader()
        
        print("\n📡 Fetching gemeente geometries from PDOK...")
        gdf = geo_loader.get_gemeenten(use_cache=True)
        
        # Calculate NAP heights
        df = self.calculate_nap_heights_for_regions(
            gdf=gdf,
            code_column='statcode',
            name_column='statnaam',
            coverage='dtm_05m',
            delay=0.2  # Be extra nice for 300+ requests
        )
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"\n💾 Cached to: {cache_file}")
        
        return df
    
    def load_or_calculate_wijk_nap(
        self,
        gemeente_code: Optional[str] = None,
        use_cache: bool = True,
        force_recalculate: bool = False
    ) -> pd.DataFrame:
        """
        Load or calculate NAP heights for wijken
        
        Args:
            gemeente_code: Optionally filter to specific gemeente (e.g., 'GM0363')
            use_cache: Try to load from cache first
            force_recalculate: Force recalculation even if cache exists
            
        Returns:
            DataFrame with wijk NAP heights
        """
        cache_suffix = gemeente_code or 'all'
        cache_file = self.cache_dir / f'wijk_nap_heights_{cache_suffix}.csv'
        
        # Try cache first
        if use_cache and not force_recalculate and cache_file.exists():
            print(f"📂 Loading wijk NAP heights from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            print(f"   ✅ Loaded {len(df)} wijken")
            return df
        
        # Load geometries
        from src.pdok_geo_loader import PDOKGeoLoader
        geo_loader = PDOKGeoLoader()
        
        print(f"\n📡 Fetching wijk geometries from PDOK{f' for {gemeente_code}' if gemeente_code else ''}...")
        gdf = geo_loader.get_wijken(gemeente_code=gemeente_code, use_cache=True)
        
        # Calculate NAP heights
        df = self.calculate_nap_heights_for_regions(
            gdf=gdf,
            code_column='statcode',
            name_column='statnaam',
            coverage='dtm_05m',
            delay=0.1
        )
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"\n💾 Cached to: {cache_file}")
        
        return df
    
    def load_or_calculate_buurt_nap(
        self,
        gemeente_code: Optional[str] = None,
        use_cache: bool = True,
        force_recalculate: bool = False,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Load or calculate NAP heights for buurten
        
        Args:
            gemeente_code: Optionally filter to specific gemeente
            use_cache: Try to load from cache first
            force_recalculate: Force recalculation even if cache exists
            batch_size: Number of buurten to process before saving checkpoint (for resumability)
            
        Returns:
            DataFrame with buurt NAP heights
        """
        cache_suffix = gemeente_code or 'all'
        cache_file = self.cache_dir / f'buurt_nap_heights_{cache_suffix}.csv'
        
        # Try cache first
        if use_cache and not force_recalculate and cache_file.exists():
            print(f"📂 Loading buurt NAP heights from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            print(f"   ✅ Loaded {len(df)} buurten")
            return df
        
        # Load geometries
        from src.pdok_geo_loader import PDOKGeoLoader
        geo_loader = PDOKGeoLoader()
        
        print(f"\n📡 Fetching buurt geometries from PDOK{f' for {gemeente_code}' if gemeente_code else ''}...")
        gdf = geo_loader.get_buurten(gemeente_code=gemeente_code, use_cache=True)
        
        # For large datasets (>1000 buurten), process in batches and save checkpoints
        if len(gdf) > 1000:
            print(f"\n⚠️  Large dataset detected ({len(gdf)} buurten)")
            print(f"   Processing in batches of {batch_size} with checkpoints")
            
            all_results = []
            checkpoint_file = self.cache_dir / f'buurt_nap_heights_{cache_suffix}_checkpoint.csv'
            
            # Resume from checkpoint if exists
            start_idx = 0
            if checkpoint_file.exists() and not force_recalculate:
                print(f"   📥 Resuming from checkpoint...")
                df_checkpoint = pd.read_csv(checkpoint_file)
                all_results.append(df_checkpoint)
                start_idx = len(df_checkpoint)
                print(f"   ✅ Loaded {start_idx} previously processed buurten")
            
            # Process remaining buurten in batches
            for batch_start in range(start_idx, len(gdf), batch_size):
                batch_end = min(batch_start + batch_size, len(gdf))
                gdf_batch = gdf.iloc[batch_start:batch_end]
                
                print(f"\n   📦 Batch {batch_start}-{batch_end} of {len(gdf)}...")
                
                df_batch = self.calculate_nap_heights_for_regions(
                    gdf=gdf_batch,
                    code_column='statcode',
                    name_column='statnaam',
                    coverage='dtm_05m',
                    delay=0.05
                )
                
                if len(df_batch) > 0:
                    all_results.append(df_batch)
                    
                    # Save checkpoint
                    df_combined = pd.concat(all_results, ignore_index=True)
                    df_combined.to_csv(checkpoint_file, index=False)
                    print(f"   💾 Checkpoint saved: {len(df_combined)} buurten processed")
            
            # Final combine
            df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
            
            # Save final result
            df.to_csv(cache_file, index=False)
            print(f"\n💾 Final cache saved: {cache_file}")
            
            # Remove checkpoint file
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                print(f"   🗑️  Checkpoint file removed")
        else:
            # Small dataset, process all at once
            df = self.calculate_nap_heights_for_regions(
                gdf=gdf,
                code_column='statcode',
                name_column='statnaam',
                coverage='dtm_05m',
                delay=0.05
            )
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            print(f"\n💾 Cached to: {cache_file}")
        
        return df
    
    def load_all_nap_heights(
        self,
        use_cache: bool = True,
        force_recalculate: bool = False
    ) -> pd.DataFrame:
        """
        Load NAP heights for all geographic levels (gemeente, wijk, buurt)
        
        Returns:
            Combined DataFrame with all NAP heights
        """
        print("\n" + "=" * 70)
        print("🏔️  LOADING NAP HEIGHTS FOR ALL LEVELS")
        print("=" * 70)
        
        dfs = []
        
        # Gemeenten
        print("\n1️⃣  GEMEENTEN")
        df_gem = self.load_or_calculate_gemeente_nap(
            use_cache=use_cache,
            force_recalculate=force_recalculate
        )
        if len(df_gem) > 0:
            dfs.append(df_gem)
        
        # Wijken
        print("\n2️⃣  WIJKEN")
        df_wijk = self.load_or_calculate_wijk_nap(
            use_cache=use_cache,
            force_recalculate=force_recalculate
        )
        if len(df_wijk) > 0:
            dfs.append(df_wijk)
        
        # Buurten
        print("\n3️⃣  BUURTEN")
        df_buurt = self.load_or_calculate_buurt_nap(
            use_cache=use_cache,
            force_recalculate=force_recalculate
        )
        if len(df_buurt) > 0:
            dfs.append(df_buurt)
        
        # Combine all
        if not dfs:
            raise ValueError("No NAP data loaded!")
        
        df_combined = pd.concat(dfs, ignore_index=True)
        
        print("\n" + "=" * 70)
        print("✅ COMBINED NAP HEIGHTS")
        print(f"   Total regions: {len(df_combined):,}")
        print(f"   Gemeenten: {len([c for c in df_combined['gwb_code_10'] if c.startswith('GM')]):,}")
        print(f"   Wijken: {len([c for c in df_combined['gwb_code_10'] if c.startswith('WK')]):,}")
        print(f"   Buurten: {len([c for c in df_combined['gwb_code_10'] if c.startswith('BU')]):,}")
        print(f"   NAP range: {df_combined['nap_hoogte_gem'].min():.2f}m - {df_combined['nap_hoogte_gem'].max():.2f}m")
        print(f"   NAP mean: {df_combined['nap_hoogte_gem'].mean():.2f}m")
        print("=" * 70)
        
        return df_combined


def load_nap_heights_for_dashboard(use_cache: bool = True) -> pd.DataFrame:
    """
    Convenience function to load NAP heights for dashboard integration
    
    Returns:
        DataFrame with columns: gwb_code_10, nap_hoogte_gem
    """
    loader = AHNNAPLoader()
    df = loader.load_all_nap_heights(use_cache=use_cache)
    
    # Return only essential columns for merging
    return df[['gwb_code_10', 'nap_hoogte_gem']]


if __name__ == '__main__':
    print("🧪 TESTING AHN NAP LOADER\n")
    
    # Test with small sample first
    from src.pdok_geo_loader import PDOKGeoLoader
    
    geo_loader = PDOKGeoLoader()
    loader = AHNNAPLoader()
    
    # Test 1: Single gemeente (Amsterdam)
    print("\n" + "=" * 70)
    print("TEST 1: Single gemeente (Amsterdam)")
    print("=" * 70)
    
    gdf_adam = geo_loader.get_gemeenten(use_cache=True)
    gdf_adam = gdf_adam[gdf_adam['statcode'] == 'GM0363']
    
    df_test = loader.calculate_nap_heights_for_regions(
        gdf=gdf_adam,
        code_column='statcode',
        name_column='statnaam',
        max_regions=1
    )
    
    if len(df_test) > 0:
        print(f"\n✅ Test successful!")
        print(df_test)
    else:
        print(f"\n❌ Test failed - no results")
    
    # Test 2: Few gemeenten
    print("\n" + "=" * 70)
    print("TEST 2: First 3 gemeenten")
    print("=" * 70)
    
    gdf_sample = geo_loader.get_gemeenten(use_cache=True).head(3)
    df_sample = loader.calculate_nap_heights_for_regions(
        gdf=gdf_sample,
        code_column='statcode',
        name_column='statnaam'
    )
    
    print("\n📊 RESULTS:")
    print(df_sample.to_string(index=False))
