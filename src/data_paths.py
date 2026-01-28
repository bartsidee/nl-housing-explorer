"""
Data Paths Configuration
Centralized paths for all data files to support new folder structure
"""
from pathlib import Path

# Base directories
DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
GEO_DIR = DATA_DIR / 'geo'
EXPORTS_DIR = DATA_DIR / 'exports'

# Raw data paths
KWB_RAW_DIR = RAW_DIR / 'kwb'
SES_RAW_DIR = RAW_DIR / 'ses' / '85900NED'

def get_kwb_path(year: int) -> Path:
    """Get path to KWB Excel file for given year"""
    return KWB_RAW_DIR / str(year) / f'kwb-{year}.xlsx'

# Processed data paths
CURRENT_DATA_DIR = PROCESSED_DIR / 'current'
MAIN_DATA_PATH = CURRENT_DATA_DIR / 'main_data.csv'
ARCHIVE_DIR = PROCESSED_DIR / 'archive'
SES_PROCESSED_DIR = PROCESSED_DIR / 'ses'

def get_processed_path(year: int) -> Path:
    """Get path to processed KWB data for specific year"""
    return PROCESSED_DIR / str(year) / 'main_data.csv'

# Geo cache paths
GEO_CACHE_DIR = GEO_DIR / 'cache'

# Current data path (used in production)
MAIN_DATA_WITH_TRENDS_PATH = CURRENT_DATA_DIR / 'main_data_with_trends.csv'
METADATA_PATH = CURRENT_DATA_DIR / 'metadata.json'

# SES data files
SES_OBSERVATIONS = SES_RAW_DIR / 'Observations.csv'
SES_DIMENSIONS = SES_RAW_DIR / 'Dimensions.csv'
SES_WIJKEN_CODES = SES_RAW_DIR / 'WijkenEnBuurtenCodes.csv'
