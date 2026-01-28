#!/usr/bin/env python3
"""
FAST NAP Data Downloader - Uses parallel processing

⚠️ WARNING: Uses 4-8 parallel workers
- 4x-8x faster than sequential
- Higher load on PDOK API
- Use responsibly!

Usage:
    python scripts/download_nap_data_fast.py --level gemeente --workers 4
    python scripts/download_nap_data_fast.py --level all --workers 4
"""

import sys
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ahn_nap_loader import AHNNAPLoader
from src.pdok_geo_loader import PDOKGeoLoader


def process_region_safe(args):
    """
    Wrapper to safely process a single region
    Returns: (success, result_dict or None)
    """
    loader, row, coverage = args
    
    try:
        code = row['statcode']
        name = row.get('statnaam', 'Unknown')
        geometry = row.geometry
        
        # Calculate NAP height
        mean_elevation = loader._calculate_mean_elevation_for_geometry(
            geometry=geometry,
            coverage=coverage
        )
        
        if mean_elevation is not None:
            return (True, {
                'gwb_code_10': code,
                'region_name': name,
                'nap_hoogte_gem': round(mean_elevation, 2)
            })
        else:
            return (False, None)
            
    except Exception as e:
        return (False, None)


def download_level_parallel(
    level: str,
    workers: int = 4,
    use_cache: bool = True
):
    """
    Download NAP data for a level using parallel processing
    
    Args:
        level: 'gemeente', 'wijk', or 'buurt'
        workers: Number of parallel workers (4-8 recommended)
        use_cache: Load from cache if exists
    """
    print(f"\n{'=' * 70}")
    print(f"🏔️  FAST DOWNLOAD: {level.upper()} (with {workers} workers)")
    print(f"{'=' * 70}")
    
    loader = AHNNAPLoader()
    geo_loader = PDOKGeoLoader()
    
    # Check cache
    if level == 'gemeente':
        cache_file = loader.cache_dir / 'gemeente_nap_heights.csv'
    elif level == 'wijk':
        cache_file = loader.cache_dir / 'wijk_nap_heights_all.csv'
    else:  # buurt
        cache_file = loader.cache_dir / 'buurt_nap_heights_all.csv'
    
    if use_cache and cache_file.exists():
        print(f"✅ Cache exists: {cache_file}")
        df = pd.read_csv(cache_file)
        print(f"   Loaded {len(df)} records from cache")
        return df
    
    # Load geometries
    print(f"\n📡 Fetching {level} geometries from PDOK...")
    if level == 'gemeente':
        gdf = geo_loader.get_gemeenten(use_cache=True)
    elif level == 'wijk':
        gdf = geo_loader.get_wijken(use_cache=True)
    else:
        gdf = geo_loader.get_buurten(use_cache=True)
    
    print(f"   ✅ Loaded {len(gdf)} {level} geometries")
    
    # Prepare tasks
    print(f"\n⚡ Processing with {workers} parallel workers...")
    tasks = [(loader, row, 'dtm_05m') for idx, row in gdf.iterrows()]
    
    results = []
    failed = 0
    completed = 0
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_region_safe, task): i 
                  for i, task in enumerate(tasks)}
        
        for future in as_completed(futures):
            success, result = future.result()
            completed += 1
            
            if success:
                results.append(result)
            else:
                failed += 1
            
            # Progress update every 50 regions
            if completed % 50 == 0:
                print(f"   Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.0f}%)")
    
    print(f"\n{'=' * 70}")
    print(f"✅ Processed: {len(results)}/{len(tasks)} regions")
    print(f"   Success: {len(results)}")
    print(f"   Failed: {failed}")
    
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        # Statistics
        print(f"\n📊 NAP Statistics:")
        print(f"   Range: {df['nap_hoogte_gem'].min():.2f}m - {df['nap_hoogte_gem'].max():.2f}m")
        print(f"   Mean: {df['nap_hoogte_gem'].mean():.2f}m")
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"\n💾 Saved to: {cache_file}")
        
        return df
    else:
        print(f"\n❌ No data downloaded!")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Fast NAP data downloader with parallel processing'
    )
    parser.add_argument(
        '--level',
        choices=['gemeente', 'wijk', 'buurt', 'all'],
        default='gemeente',
        help='Geographic level to download'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, max recommended: 8)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force redownload (ignore cache)'
    )
    
    args = parser.parse_args()
    
    # Validate workers
    if args.workers > 8:
        print(f"⚠️  Warning: {args.workers} workers is high!")
        print(f"   Recommended max: 8 workers")
        print(f"   Risk: PDOK API throttling")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)
    
    print("\n" + "=" * 70)
    print("🏔️  FAST NAP DATA DOWNLOADER")
    print("=" * 70)
    print(f"   Level: {args.level}")
    print(f"   Workers: {args.workers} (parallel)")
    print(f"   Force: {args.force}")
    print(f"\n   ⚡ Estimated time with {args.workers} workers:")
    
    times = {
        'gemeente': 342 * 0.5 / args.workers,
        'wijk': 3300 * 0.5 / args.workers,
        'buurt': 14500 * 0.5 / args.workers
    }
    
    if args.level == 'all':
        total_time = sum(times.values())
        print(f"      Gemeenten: ~{times['gemeente']/60:.0f} min")
        print(f"      Wijken: ~{times['wijk']/60:.0f} min")
        print(f"      Buurten: ~{times['buurt']/60:.0f} min")
        print(f"      Total: ~{total_time/60:.0f} min")
    else:
        print(f"      {args.level.capitalize()}: ~{times[args.level]/60:.0f} min")
    
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        if args.level in ['gemeente', 'all']:
            download_level_parallel('gemeente', args.workers, not args.force)
        
        if args.level in ['wijk', 'all']:
            download_level_parallel('wijk', args.workers, not args.force)
        
        if args.level in ['buurt', 'all']:
            download_level_parallel('buurt', args.workers, not args.force)
        
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"✅ DOWNLOAD COMPLETE!")
        print(f"   Total time: {elapsed/60:.1f} minutes")
        print(f"   Speedup: ~{args.workers}x vs sequential")
        print(f"{'=' * 70}")
        
        print("\nNext steps:")
        print("1. git add data/geo/cache/ahn/*.csv")
        print("2. git commit -m 'Add NAP heights'")
        print("3. python scripts/process_multiyear_trends.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
