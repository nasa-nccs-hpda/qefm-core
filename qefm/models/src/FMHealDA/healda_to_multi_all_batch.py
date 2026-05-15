#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
HealDA Multi-Model Weather Prediction Pipeline
===============================================

This script demonstrates a comprehensive weather prediction pipeline supporting
all Earth2Studio forecast models with intelligent caching and fine-grained control.

Features:
- Automatic detection and loading of cached HealDA analysis and ERA5 data
- Support for both UFS and MERRA-21c observation sources
- Parameterized observation time windows
- Optional data-only mode (fetch and cache without analysis)
- Selective model execution
- Support for 23+ forecast models
- Comprehensive metrics and visualization
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import torch
import xarray as xr
from datetime import timedelta
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import importlib

# Configure logger
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

# Earth2Studio imports
from earth2studio.data import fetch_dataframe, NCAR_ERA5
from earth2studio.models.da import HealDA
from earth2studio.io import ZarrBackend
import sys
import os

# Add parent directory to path (common use case)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

print("Updated sys.path:")
print("\n".join(sys.path))

# Now you can import from the parent directory
# from parent_module import SomeClass
from merra21c import MERRA21cObsConv, MERRA21cObsSat
print("from ext.merra21c import MERRA21cObsConv, MERRA21cObsSat")


# Import both UFS and MERRA-21c observation sources
try:
    from earth2studio.data import UFSObsConv, UFSObsSat
    UFS_AVAILABLE = True
except ImportError:
    logger.warning("UFS observation sources not available")
    UFS_AVAILABLE = False

try:
    from merra21c import MERRA21cObsConv, MERRA21cObsSat
    MERRA21C_AVAILABLE = True
except ImportError:
    logger.warning("MERRA-21c observation sources not available")
    MERRA21C_AVAILABLE = False

# %%
# Model Registry
# --------------

MODEL_REGISTRY = {
    # Global AI Weather Models
    'ace2era5': {
        'name': 'ACE2ERA5',
        'module': 'earth2studio.models.px',
        'class': 'ACE2ERA5',
        'timestep': 6,
        'description': 'ACE2 trained on ERA5 data'
    },
    'aifs': {
        'name': 'AIFS',
        'module': 'earth2studio.models.px',
        'class': 'AIFS',
        'timestep': 6,
        'description': 'ECMWF AI Integrated Forecasting System'
    },
    'aifsens': {
        'name': 'AIFSENS',
        'module': 'earth2studio.models.px',
        'class': 'AIFSENS',
        'timestep': 6,
        'description': 'AIFS Ensemble version'
    },
    'atlas': {
        'name': 'Atlas',
        'module': 'earth2studio.models.px',
        'class': 'Atlas',
        'timestep': 6,
        'description': 'Atlas weather model'
    },
    'aurora': {
        'name': 'Aurora',
        'module': 'earth2studio.models.px',
        'class': 'Aurora',
        'timestep': 6,
        'description': 'Microsoft Aurora'
    },
    'fengwu': {
        'name': 'FengWu',
        'module': 'earth2studio.models.px',
        'class': 'FengWu',
        'timestep': 6,
        'description': 'FengWu global weather model'
    },
    'fuxi': {
        'name': 'FuXi',
        'module': 'earth2studio.models.px',
        'class': 'FuXi',
        'timestep': 6,
        'description': 'FuXi weather forecasting model'
    },
    'graphcast': {
        'name': 'GraphCastOperational',
        'module': 'earth2studio.models.px',
        'class': 'GraphCastOperational',
        'timestep': 6,
        'description': 'GraphCast operational (high-res)'
    },
    'graphcast-small': {
        'name': 'GraphCastSmall',
        'module': 'earth2studio.models.px',
        'class': 'GraphCastSmall',
        'timestep': 6,
        'description': 'GraphCast small (0.25 degree)'
    },
    'pangu24': {
        'name': 'Pangu24',
        'module': 'earth2studio.models.px',
        'class': 'Pangu24',
        'timestep': 24,
        'description': 'Pangu-Weather 24-hour model'
    },
    'pangu6': {
        'name': 'Pangu6',
        'module': 'earth2studio.models.px',
        'class': 'Pangu6',
        'timestep': 6,
        'description': 'Pangu-Weather 6-hour model'
    },
    'pangu3': {
        'name': 'Pangu3',
        'module': 'earth2studio.models.px',
        'class': 'Pangu3',
        'timestep': 3,
        'description': 'Pangu-Weather 3-hour model'
    },
    'sfno': {
        'name': 'SFNO',
        'module': 'earth2studio.models.px',
        'class': 'SFNO',
        'timestep': 6,
        'description': 'Spherical Fourier Neural Operator'
    },
    
    # Regional/Specialized Models
    'fcn': {
        'name': 'FCN',
        'module': 'earth2studio.models.px',
        'class': 'FCN',
        'timestep': 6,
        'description': 'FourCastNet'
    },
    'fcn3': {
        'name': 'FCN3',
        'module': 'earth2studio.models.px',
        'class': 'FCN3',
        'timestep': 6,
        'description': 'FourCastNet version 3'
    },
    'dlwp': {
        'name': 'DLWP',
        'module': 'earth2studio.models.px',
        'class': 'DLWP',
        'timestep': 6,
        'description': 'Deep Learning Weather Prediction'
    },
    'dlesym': {
        'name': 'DLESyM',
        'module': 'earth2studio.models.px',
        'class': 'DLESyM',
        'timestep': 6,
        'description': 'DLESyM on cubed sphere'
    },
    'dlesym-latlon': {
        'name': 'DLESyMLatLon',
        'module': 'earth2studio.models.px',
        'class': 'DLESyMLatLon',
        'timestep': 6,
        'description': 'DLESyM on lat-lon grid'
    },
    'interpmodafno': {
        'name': 'InterpModAFNO',
        'module': 'earth2studio.models.px',
        'class': 'InterpModAFNO',
        'timestep': 6,
        'description': 'Interpolated Modulus AFNO'
    },
    
    # Storm/Precipitation Models
    'stormcast': {
        'name': 'StormCast',
        'module': 'earth2studio.models.px',
        'class': 'StormCast',
        'timestep': 1,
        'description': 'StormCast precipitation nowcasting'
    },
    'stormscope-goes': {
        'name': 'StormScopeGOES',
        'module': 'earth2studio.models.px',
        'class': 'StormScopeGOES',
        'timestep': 1,
        'description': 'StormScope using GOES satellite'
    },
    'stormscope-mrms': {
        'name': 'StormScopeMRMS',
        'module': 'earth2studio.models.px',
        'class': 'StormScopeMRMS',
        'timestep': 1,
        'description': 'StormScope using MRMS radar'
    },
    
    # Video/Time Series Models
    'cbottlevideo': {
        'name': 'CBottleVideo',
        'module': 'earth2studio.models.px',
        'class': 'CBottleVideo',
        'timestep': 6,
        'description': 'CorrDiff Bottleneck Video model'
    },
    
    # Utility Models
    'diagnosticwrapper': {
        'name': 'DiagnosticWrapper',
        'module': 'earth2studio.models.px',
        'class': 'DiagnosticWrapper',
        'timestep': 6,
        'description': 'Wrapper for diagnostic variables'
    },
    'persistence': {
        'name': 'Persistence',
        'module': 'earth2studio.models.px',
        'class': 'Persistence',
        'timestep': 6,
        'description': 'Persistence baseline forecast'
    }
}

# %%
# Utility Functions
# -----------------

def to_numpy(arr):
    """Convert CuPy or NumPy array to NumPy array"""
    if hasattr(arr, 'get'):
        return arr.get()  # CuPy array
    else:
        return np.asarray(arr)  # NumPy array

def load_model_class(module_name, class_name):
    """Dynamically load a model class"""
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load {class_name} from {module_name}: {e}")

def load_and_initialize_model(model_key, device):
    """Load and initialize a forecast model"""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")
    
    config = MODEL_REGISTRY[model_key]
    logger.info(f"Loading {config['name']}...")
    logger.info(f"  Description: {config['description']}")
    
    try:
        # Load model class
        ModelClass = load_model_class(config['module'], config['class'])
        
        # Load model package and weights
        package = ModelClass.load_default_package()
        model = ModelClass.load_model(package)
        model = model.to(device)
        
        logger.info(f"  ✓ {config['name']} loaded successfully")
        return model, config
    
    except Exception as e:
        logger.error(f"  ✗ Failed to load {config['name']}: {e}")
        raise

def safe_forecast(model, initial_state, model_name, device, forecast_hours, timestep):
    """Safely run forecast model with error handling"""
    try:
        logger.info(f"  Running {model_name} inference...")
        n_steps = forecast_hours // timestep
        logger.info(f"  Forecast steps: {n_steps} (every {timestep} hours)")
        
        # Different models may have different call signatures
        try:
            # Try standard call
            forecast = model(
                x=initial_state,
                normalize=True,
            )
        except TypeError:
            # Try without normalize
            try:
                forecast = model(x=initial_state)
            except:
                # Try as direct call
                forecast = model(initial_state)
        
        logger.info(f"  ✓ {model_name} forecast shape: {forecast.shape}")
        return forecast, None
    
    except Exception as e:
        error_msg = f"Error running {model_name}: {str(e)}"
        logger.error(f"  ✗ {error_msg}")
        return None, error_msg

def load_cached_data(output_dir, analysis_time):
    """Check for and load cached HealDA analysis and ERA5 validation data"""
    output_path = Path(output_dir)
    
    # Check for HealDA analysis
    healda_file = output_path / "healda_analysis.nc"
    healda_data = None
    if healda_file.exists():
        try:
            healda_data = xr.open_dataarray(healda_file)
            # Verify the time matches
            if healda_data.coords['time'].values[0] == analysis_time[0]:
                logger.info(f"✓ Found cached HealDA analysis: {healda_file}")
            else:
                logger.warning(f"Cached HealDA analysis time mismatch, will regenerate")
                healda_data = None
        except Exception as e:
            logger.warning(f"Failed to load cached HealDA analysis: {e}")
            healda_data = None
    
    # Check for ERA5 validation
    era5_file = output_path / "era5_validation.nc"
    era5_data = None
    if era5_file.exists():
        try:
            era5_data = xr.open_dataarray(era5_file)
            # Verify the time includes analysis_time
            if analysis_time[0] in era5_data.coords['time'].values:
                logger.info(f"✓ Found cached ERA5 validation: {era5_file}")
            else:
                logger.warning(f"Cached ERA5 validation time mismatch, will regenerate")
                era5_data = None
        except Exception as e:
            logger.warning(f"Failed to load cached ERA5 validation: {e}")
            era5_data = None
    
    return healda_data, era5_data

# %%
# Parse Command Line Arguments
# -----------------------------

def list_models():
    """Print available models and exit"""
    print("\n" + "="*80)
    print("AVAILABLE FORECAST MODELS")
    print("="*80)
    
    categories = {
        'Global AI Weather Models': [
            'ace2era5', 'aifs', 'aifsens', 'atlas', 'aurora', 'fengwu', 'fuxi',
            'graphcast', 'graphcast-small', 'pangu24', 'pangu6', 'pangu3', 'sfno'
        ],
        'Regional/Specialized Models': [
            'fcn', 'fcn3', 'dlwp', 'dlesym', 'dlesym-latlon', 'interpmodafno'
        ],
        'Storm/Precipitation Models': [
            'stormcast', 'stormscope-goes', 'stormscope-mrms'
        ],
        'Video/Time Series Models': [
            'cbottlevideo'
        ],
        'Utility Models': [
            'diagnosticwrapper', 'persistence'
        ]
    }
    
    for category, models in categories.items():
        print(f"\n{category}:")
        for model_key in models:
            if model_key in MODEL_REGISTRY:
                config = MODEL_REGISTRY[model_key]
                print(f"  {model_key:20s} - {config['description']}")
    
    print("\n" + "="*80)
    print(f"Total: {len(MODEL_REGISTRY)} models available")
    print("="*80 + "\n")

parser = argparse.ArgumentParser(
    description='HealDA multi-model weather prediction pipeline with intelligent caching',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # List all available models
  python script.py --list-models
  
  # Fetch and cache UFS data only
  python script.py --data-only
  
  # Fetch and cache MERRA-21c data only
  python script.py --use-merra21c --data-only
  
  # Run HealDA analysis with MERRA-21c observations
  python script.py --use-merra21c --skip-forecast
  
  # Run specific forecast models with MERRA-21c
  python script.py --use-merra21c --models graphcast aurora pangu6
  
  # Custom observation time window with MERRA-21c
  python script.py --use-merra21c --obs-window-start -21 --obs-window-end 3
  
  # Complete workflow with custom settings
  python script.py --use-merra21c --models graphcast aurora --device cuda:0 --forecast-hours 240
    """
)

parser.add_argument('--list-models', action='store_true',
                    help='List all available models and exit')
parser.add_argument('--analysis-time', type=str, default='2024-01-01T00:00',
                    help='Analysis time in ISO format (default: 2024-01-01T00:00)')
parser.add_argument('--forecast-hours', type=int, default=120,
                    help='Forecast length in hours (default: 120)')
parser.add_argument('--models', type=str, nargs='+',
                    default=None,
                    help='Forecast models to run (use --list-models to see options). If not specified, no forecasts will be run.')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to run models on (default: cpu, or cuda:0, cuda:1, etc.)')
parser.add_argument('--output-dir', type=str, default='outputs/healda_multimodel',
                    help='Output directory for results')
parser.add_argument('--model-cache', type=str, default=None,
                    help='Path to local model cache directory (optional)')

# Observation source selection
parser.add_argument('--use-merra21c', action='store_true',
                    help='Use MERRA-21c observations instead of UFS (default: UFS)')
parser.add_argument('--merra21c-base-path', type=str,
                    default='/discover/nobackup/projects/gmao/merra21c/TSE_staging',
                    help='Base path for MERRA-21c data (default: /discover/nobackup/projects/gmao/merra21c/TSE_staging)')
parser.add_argument('--merra21c-experiment', type=str,
                    default='e5303_m21c_jan18',
                    help='MERRA-21c experiment ID (default: e5303_m21c_jan18)')

# Data fetching control
parser.add_argument('--obs-window-start', type=int, default=-21,
                    help='Observation window start in hours relative to analysis time (default: -21)')
parser.add_argument('--obs-window-end', type=int, default=3,
                    help='Observation window end in hours relative to analysis time (default: 3)')
parser.add_argument('--data-only', action='store_true',
                    help='Only fetch and save observations/ERA5 data, then exit (no analysis or forecasts)')

# Processing control
parser.add_argument('--skip-healda', action='store_true',
                    help='Skip HealDA and use ERA5 as initial conditions')
parser.add_argument('--skip-era5', action='store_true',
                    help='Skip ERA5 validation data fetching')
parser.add_argument('--skip-forecast', action='store_true',
                    help='Skip forecast model execution (only run HealDA analysis)')
parser.add_argument('--skip-plots', action='store_true',
                    help='Skip generating visualization plots')

# Cache control
parser.add_argument('--force-healda', action='store_true',
                    help='Force regenerate HealDA analysis (ignore cache)')
parser.add_argument('--force-era5', action='store_true',
                    help='Force regenerate ERA5 validation (ignore cache)')

args = parser.parse_args()

# Handle --list-models
if args.list_models:
    list_models()
    exit(0)

# Validate observation source availability
if args.use_merra21c and not MERRA21C_AVAILABLE:
    logger.error("MERRA-21c observation sources not available. Check if merra21c.py is in your path.")
    exit(1)

if not args.use_merra21c and not UFS_AVAILABLE:
    logger.error("UFS observation sources not available. Use --use-merra21c or install earth2studio with UFS support.")
    exit(1)

# Validate model selection if models specified
if args.models:
    invalid_models = [m for m in args.models if m not in MODEL_REGISTRY]
    if invalid_models:
        logger.error(f"Invalid model(s): {', '.join(invalid_models)}")
        logger.info("Use --list-models to see available models")
        exit(1)

# Set skip-forecast if no models specified and not data-only mode
if not args.models and not args.data_only:
    args.skip_forecast = True
    logger.info("No models specified, will skip forecast execution")

# %%
# Setup
# -----

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

analysis_time = np.array([np.datetime64(args.analysis_time)])
obs_source_name = "MERRA-21c" if args.use_merra21c else "UFS"

logger.info("=" * 80)
logger.info("HEALDA MULTI-MODEL WEATHER PREDICTION PIPELINE")
logger.info("=" * 80)
logger.info(f"Observation source: {obs_source_name}")
if args.use_merra21c:
    logger.info(f"  Base path: {args.merra21c_base_path}")
    logger.info(f"  Experiment: {args.merra21c_experiment}")
logger.info(f"Analysis time: {analysis_time[0]}")
logger.info(f"Observation window: {args.obs_window_start}h to {args.obs_window_end}h")

if args.data_only:
    logger.info(f"Mode: DATA ONLY (fetch and cache, then exit)")
elif args.skip_forecast:
    logger.info(f"Mode: ANALYSIS ONLY (no forecast models)")
else:
    logger.info(f"Forecast length: {args.forecast_hours} hours")
    logger.info(f"Models to run: {', '.join(args.models)}")

logger.info(f"Device: {args.device}")
logger.info(f"Output directory: {output_dir}")

# Set model cache if provided
if args.model_cache:
    os.environ['EARTH2STUDIO_CACHE'] = args.model_cache
    logger.info(f"Using model cache directory: {args.model_cache}")

# %%
# Check for Cached Data
# ---------------------

logger.info("=" * 80)
logger.info("CHECKING FOR CACHED DATA")
logger.info("=" * 80)

cached_healda, cached_era5 = load_cached_data(output_dir, analysis_time)

use_cached_healda = cached_healda is not None and not args.force_healda and not args.skip_healda
use_cached_era5 = cached_era5 is not None and not args.force_era5 and not args.skip_era5

if use_cached_healda:
    logger.info("Will use cached HealDA analysis")
else:
    if args.force_healda:
        logger.info("Force regenerate HealDA analysis (--force-healda)")
    elif args.skip_healda:
        logger.info("Skipping HealDA (--skip-healda)")
    else:
        logger.info("No valid cached HealDA analysis found")

if use_cached_era5:
    logger.info("Will use cached ERA5 validation")
else:
    if args.force_era5:
        logger.info("Force regenerate ERA5 validation (--force-era5)")
    elif args.skip_era5:
        logger.info("Skipping ERA5 validation (--skip-era5)")
    else:
        logger.info("No valid cached ERA5 validation found")

# %%
# Step 1: Generate Initial Conditions (HealDA or ERA5)
# -----------------------------------------------------

# Define observation time tolerance
obs_time_tolerance = (
    timedelta(hours=args.obs_window_start),
    timedelta(hours=args.obs_window_end)
)
logger.info(f"Using observation time tolerance: {obs_time_tolerance}")

if use_cached_healda:
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Cached HealDA Analysis")
    logger.info("=" * 80)
    
    initial_state = cached_healda
    logger.info(f"✓ Loaded HealDA analysis shape: {initial_state.shape}")
    ic_source = "HealDA (cached)"

elif not args.skip_healda:
    logger.info("=" * 80)
    logger.info(f"STEP 1: HealDA Data Assimilation ({obs_source_name} observations)")
    logger.info("=" * 80)
    
    try:
        # Load HealDA Model (skip if data-only mode)
        if not args.data_only:
            logger.info("Loading HealDA model...")
            healda_package = HealDA.load_default_package()
            healda_model = HealDA.load_model(healda_package, lat_lon=True)
            healda_model = healda_model.to(args.device)
            logger.info(f"✓ HealDA model loaded on device: {args.device}")
            conv_schema, sat_schema = healda_model.input_coords()
        else:
            # In data-only mode, use default schema
            logger.info("Data-only mode: using default observation schema")

            # Specify conventional variables and satellite instruments of interest here
            if args.use_merra21c:
                # MERRA-21c schema
                conv_schema = {
                    'variable': ['t', 'q', 'u', 'v'],
                    'time': None, 'pres': None, 'elev': None, 'type': None,
                    'class': None, 'lat': None, 'lon': None, 'station': None,
                    'station_elev': None, 'observation': None
                }
                # Use actual satellite instruments, not 'brightness_temperature'
                sat_schema = {
                    'variable': ['atms', 'amsua', 'iasi'],  # Actual satellite instruments
                    'time': None, 'lat': None, 'lon': None, 'channel_index': None,
                    'observation': None, 'elev': None, 'class': None,
                    'scan_angle': None, 'solza': None, 'solaza': None,
                    'satellite_za': None, 'satellite_aza': None, 'satellite': None
                }
            else:
                # UFS schema
                conv_schema = {
                    'variable': ['t', 'q', 'u', 'v'],
                    'time': None, 'pres': None, 'elev': None, 'type': None,
                    'class': None, 'lat': None, 'lon': None, 'station': None,
                    'station_elev': None, 'observation': None
                }
                sat_schema = {
                    'variable': ['atms', 'amsua', 'iasi'],  # Actual satellite instruments
                    'time': None, 'lat': None, 'lon': None, 'channel': None,
                    'observation': None, 'observation_error': None,
                    'sensor': None, 'satellite': None
                }

        # Create observation sources based on selection
        logger.info(f"Creating {obs_source_name} observation sources...")
        if args.use_merra21c:
            conv_source = MERRA21cObsConv(
                base_path=args.merra21c_base_path,
                experiment_id=args.merra21c_experiment,
                time_tolerance=obs_time_tolerance
            )
            sat_source = MERRA21cObsSat(
                base_path=args.merra21c_base_path,
                experiment_id=args.merra21c_experiment,
                time_tolerance=obs_time_tolerance
            )
        else:
            conv_source = UFSObsConv(time_tolerance=obs_time_tolerance)
            sat_source = UFSObsSat(time_tolerance=obs_time_tolerance)
        
        # Fetch Observations
        logger.info("Fetching observations...")
        
        conv_df = fetch_dataframe(
            conv_source,
            time=analysis_time,
            variable=np.array(conv_schema["variable"]),
            fields=np.array(list(conv_schema.keys())),
        )
        logger.info(f"✓ Fetched {len(conv_df):,} conventional observations")
        
        # Around line 655-665, replace the satellite fetching section:

        sat_df = fetch_dataframe(
            sat_source,
            time=analysis_time,
            variable=np.array(sat_schema["variable"]),
            fields=None,  # Let the source determine the fields instead of forcing HealDA's schema
        )
        logger.info(f"✓ Fetched {len(sat_df):,} satellite observations")        
        # Save observations
        import pickle
        obs_dir = output_dir / "observations"
        obs_dir.mkdir(exist_ok=True)
        
        time_str = str(analysis_time[0]).replace(':', '-')
        obs_prefix = "merra21c" if args.use_merra21c else "ufs"
        conv_file = obs_dir / f"{obs_prefix}_conv_obs_{time_str}.pkl"
        sat_file = obs_dir / f"{obs_prefix}_sat_obs_{time_str}.pkl"
        
        with open(conv_file, 'wb') as f:
            pickle.dump(conv_df, f)
        logger.info(f"✓ Saved conventional observations to {conv_file}")
        
        with open(sat_file, 'wb') as f:
            pickle.dump(sat_df, f)
        logger.info(f"✓ Saved satellite observations to {sat_file}")
        
        # Exit if data-only mode
        if args.data_only:
            logger.info("=" * 80)
            logger.info("DATA-ONLY MODE: Observations fetched and saved")
            logger.info("=" * 80)
            logger.info("Exiting as requested (--data-only flag)")
            exit(0)
        
        # Run HealDA
        logger.info("Running HealDA data assimilation...")
        torch.manual_seed(42)
        initial_state = healda_model(conv_obs=conv_df, sat_obs=sat_df)
        logger.info(f"✓ HealDA analysis shape: {initial_state.shape}")
        
        # Save HealDA analysis
        analysis_file = output_dir / "healda_analysis.nc"
        initial_state.to_netcdf(analysis_file)
        logger.info(f"✓ Saved HealDA analysis to {analysis_file}")
        
        ic_source = f"HealDA ({obs_source_name})"
    
    except Exception as e:
        logger.error(f"✗ HealDA failed: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Falling back to ERA5 initial conditions")
        args.skip_healda = True

if args.skip_healda and not use_cached_healda:
    logger.info("=" * 80)
    logger.info("STEP 1: Using ERA5 as Initial Conditions")
    logger.info("=" * 80)
    
    # Fetch ERA5 data as initial conditions
    logger.info("Fetching ERA5 data...")
    era5_ds = NCAR_ERA5()
    
    # Define variables needed for forecast models
    # Use specific level variables instead of just 'z'
    ic_variables = ['t2m', 'u10m', 'v10m', 'msl', 'z500', 't', 'u', 'v', 'q']    

    initial_state = era5_ds(analysis_time, ic_variables)
    logger.info(f"✓ ERA5 initial state shape: {initial_state.shape}")
    
    # Save ERA5 initial state
    ic_file = output_dir / "era5_initial_conditions.nc"
    initial_state.to_netcdf(ic_file)
    logger.info(f"✓ Saved ERA5 initial conditions to {ic_file}")
    
    ic_source = "ERA5"

# %%
# Step 2: Fetch ERA5 Validation Data
# -----------------------------------

era5_forecast = None

if use_cached_era5:
    logger.info("=" * 80)
    logger.info("STEP 2: Loading Cached ERA5 Validation")
    logger.info("=" * 80)
    
    era5_forecast = cached_era5
    logger.info(f"✓ Loaded ERA5 validation shape: {era5_forecast.shape}")

elif not args.skip_era5:
    logger.info("=" * 80)
    logger.info("STEP 2: Fetching ERA5 Validation Data")
    logger.info("=" * 80)
    
    try:
        era5_ds = NCAR_ERA5()
        
        # Calculate forecast times
        if args.models:
            max_timestep = max([MODEL_REGISTRY[m]['timestep'] for m in args.models])
        else:
            max_timestep = 6  # Default
        
        forecast_times = analysis_time + np.arange(0, args.forecast_hours + max_timestep, max_timestep) * np.timedelta64(1, 'h')
        
        logger.info(f"Fetching ERA5 for {len(forecast_times)} time steps...")
        validation_vars = ['t2m', 'z500', 'u10m', 'v10m']
        
        era5_forecast = era5_ds(forecast_times, validation_vars)
        logger.info(f"✓ ERA5 validation shape: {era5_forecast.shape}")
        
        # Save ERA5 validation
        era5_file = output_dir / "era5_validation.nc"
        era5_forecast.to_netcdf(era5_file)
        logger.info(f"✓ Saved ERA5 validation to {era5_file}")
        
    except Exception as e:
        logger.error(f"✗ Error fetching ERA5 validation data: {e}")
        logger.info("Continuing without ERA5 validation")

# Exit if skip-forecast
if args.skip_forecast:
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE (--skip-forecast specified)")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nGenerated files:")
    if not use_cached_healda and not args.skip_healda:
        logger.info(f"  - healda_analysis.nc")
        obs_prefix = "merra21c" if args.use_merra21c else "ufs"
        logger.info(f"  - observations/{obs_prefix}_conv_obs_*.pkl")
        logger.info(f"  - observations/{obs_prefix}_sat_obs_*.pkl")
    if not use_cached_era5 and not args.skip_era5:
        logger.info(f"  - era5_validation.nc")
    logger.info("\nSkipping forecast model execution as requested")
    #GST exit(0)

# %%
# Step 3: Run Forecast Models
# ----------------------------

# Skip if no models specified
if not args.models or len(args.models) == 0:
    logger.info("=" * 80)
    logger.info("No forecast models specified - skipping forecast execution")
    logger.info("=" * 80)
    forecasts = {}
    errors = {}
else:
    logger.info("=" * 80)
    logger.info("STEP 3: Running Forecast Models")
    logger.info("=" * 80)

    forecasts = {}
    errors = {}

    for model_key in args.models:
        logger.info("-" * 80)
        logger.info(f"MODEL: {MODEL_REGISTRY[model_key]['name']}")
        logger.info("-" * 80)
        
        try:
            # Load model
            model, config = load_and_initialize_model(model_key, args.device)
            
            # Run forecast
            forecast, error = safe_forecast(
                model, initial_state, config['name'],
                args.device, args.forecast_hours, config['timestep']
            )
            
            if forecast is not None:
                forecasts[model_key] = forecast
                
                # Save forecast
                forecast_file = output_dir / f"{model_key}_forecast.nc"
                forecast.to_netcdf(forecast_file)
                logger.info(f"  ✓ Saved {config['name']} forecast to {forecast_file}")
            else:
                errors[model_key] = error
                
        except Exception as e:
            error_msg = f"Failed to load or run {model_key}: {str(e)}"
            logger.error(f"  ✗ {error_msg}")
            errors[model_key] = error_msg

# %%
# Step 4: Calculate Metrics
# --------------------------

if era5_forecast is not None and len(forecasts) > 0:
    logger.info("=" * 80)
    logger.info("STEP 4: Calculating Forecast Metrics")
    logger.info("=" * 80)
    
    metrics = {}
    
    for model_key, forecast in forecasts.items():
        model_name = MODEL_REGISTRY[model_key]['name']
        logger.info(f"Calculating metrics for {model_name}...")
        
        try:
            # Interpolate ERA5 to model grid if needed
            era5_interp = era5_forecast.interp(
                lat=forecast.coords['lat'],
                lon=forecast.coords['lon'],
                method='nearest'
            )
            
            # Calculate RMSE for each variable
            model_metrics = {}
            common_vars = ['t2m', 'z500', 'u10m', 'v10m']
            
            for var in common_vars:
                if var in forecast.coords['variable'].values and var in era5_interp.coords['variable'].values:
                    forecast_var = forecast.sel(variable=var).values
                    era5_var = era5_interp.sel(variable=var).values
                    
                    # Align time dimensions
                    min_time = min(forecast_var.shape[0], era5_var.shape[0])
                    forecast_var = forecast_var[:min_time]
                    era5_var = era5_var[:min_time]
                    
                    # Calculate RMSE over space for each time
                    rmse_time = np.sqrt(np.mean((forecast_var - era5_var)**2, axis=(2, 3)))
                    model_metrics[var] = {
                        'rmse_mean': float(np.mean(rmse_time)),
                        'rmse_std': float(np.std(rmse_time)),
                        'rmse_time': rmse_time.flatten().tolist()
                    }
                    
                    logger.info(f"  {var} RMSE: {model_metrics[var]['rmse_mean']:.4f} ± {model_metrics[var]['rmse_std']:.4f}")
            
            if model_metrics:
                metrics[model_key] = model_metrics
            
        except Exception as e:
            logger.error(f"  ✗ Error calculating metrics for {model_name}: {e}")
    
    # Save metrics
    if metrics:
        import json
        metrics_file = output_dir / "forecast_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Saved metrics to {metrics_file}")

# %%
# Step 5: Visualize Results
# --------------------------

if not args.skip_plots:
    logger.info("=" * 80)
    logger.info("STEP 5: Visualizing Results")
    logger.info("=" * 80)

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    # Plot initial conditions
    logger.info("Creating initial conditions plot...")
    plot_vars = ["t2m", "z500"]
    #GST plot_vars = ['t2m', 'z500', 'u10m', 'v10m']
    projection = ccrs.Robinson()

    fig, axes = plt.subplots(
        1, 2,
        subplot_kw={"projection": projection},
        figsize=(14, 5),
    )

    lat = initial_state.coords["lat"].values
    lon = initial_state.coords["lon"].values

    for idx, var in enumerate(plot_vars):
        if var not in initial_state.coords['variable'].values:
            logger.warning(f"  Variable {var} not in initial state, skipping plot")
            continue
            
        ax = axes[idx]
        field = to_numpy(initial_state.sel(variable=var).data[0])
        
        im = ax.pcolormesh(
            lon, lat, field,
            transform=ccrs.PlateCarree(),
            cmap="Spectral_r" if var == "t2m" else "viridis",
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        plt.colorbar(im, ax=ax, shrink=0.7, label=var)
        ax.set_title(f"{ic_source} Analysis: {var}", fontsize=12)

    fig.suptitle(f"Initial Conditions from {ic_source} - {str(analysis_time[0])[:16]} UTC", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "initial_conditions.png", dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Saved initial conditions plot")
    plt.close()

    # Plot forecasts comparison
    if len(forecasts) > 0 and args.models:
        logger.info("Creating forecast comparison plots...")

        # Select forecast times to plot
        forecast_hours_to_plot = [24, 72, 120]
        available_hours = [h for h in forecast_hours_to_plot if h <= args.forecast_hours]

        for fhour in available_hours:
            fig, axes = plt.subplots(
                len(args.models), len(plot_vars),
                subplot_kw={"projection": projection},
                figsize=(7 * len(plot_vars), 4 * len(args.models)),
            )
            
            if len(args.models) == 1:
                axes = axes.reshape(1, -1)
            
            for model_idx, model_key in enumerate(args.models):
                if model_key not in forecasts:
                    continue
                    
                forecast = forecasts[model_key]
                model_name = MODEL_REGISTRY[model_key]['name']
                timestep = MODEL_REGISTRY[model_key]['timestep']
                time_step = fhour // timestep
                
                for var_idx, var in enumerate(plot_vars):
                    if var not in forecast.coords['variable'].values:
                        continue
                    
                    ax = axes[model_idx, var_idx]
                    
                    # Get forecast at this time
                    if time_step < forecast.shape[0]:
                        forecast_slice = forecast.isel(time=time_step)
                        field = to_numpy(forecast_slice.sel(variable=var).data[0])
                        
                        im = ax.pcolormesh(
                            forecast.coords['lon'].values,
                            forecast.coords['lat'].values,
                            field,
                            transform=ccrs.PlateCarree(),
                            cmap="Spectral_r" if var == "t2m" else "viridis",
                        )
                        ax.coastlines(linewidth=0.5)
                        ax.gridlines(linewidth=0.3, alpha=0.5)
                        plt.colorbar(im, ax=ax, shrink=0.6)
                        
                        if model_idx == 0:
                            ax.set_title(f"{var}", fontsize=11)
                        
                        if var_idx == 0:
                            ax.text(-0.1, 0.5, model_name,
                                   transform=ax.transAxes, rotation=90,
                                   va='center', fontsize=10, fontweight='bold')
            
            fig.suptitle(f"Forecast at +{fhour}h - Init: {str(analysis_time[0])[:16]} UTC", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f"forecast_comparison_{fhour}h.png", dpi=150, bbox_inches="tight")
            logger.info(f"  ✓ Saved forecast comparison for +{fhour}h")
            plt.close()

    # Plot RMSE evolution if metrics available
    if 'metrics' in locals() and metrics:
        logger.info("Creating RMSE evolution plots...")
        
        common_vars = ['t2m', 'z500']
        fig, axes = plt.subplots(1, len(common_vars), figsize=(7 * len(common_vars), 5))
        
        if len(common_vars) == 1:
            axes = [axes]
        
        for var_idx, var in enumerate(common_vars):
            ax = axes[var_idx]
            
            for model_key, model_metrics in metrics.items():
                if var in model_metrics:
                    rmse_time = model_metrics[var]['rmse_time']
                    timestep = MODEL_REGISTRY[model_key]['timestep']
                    forecast_hours_axis = np.arange(0, len(rmse_time) * timestep, timestep)
                    
                    ax.plot(forecast_hours_axis, rmse_time,
                           marker='o', label=MODEL_REGISTRY[model_key]['name'], alpha=0.7)
            
            ax.set_xlabel('Forecast Hour', fontsize=11)
            ax.set_ylabel('RMSE', fontsize=11)
            ax.set_title(f'{var} RMSE vs ERA5', fontsize=12)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"Forecast Error Evolution - Init: {str(analysis_time[0])[:16]} UTC", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "rmse_evolution.png", dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Saved RMSE evolution plot")
        plt.close()

# %%
# Summary
# -------

logger.info("=" * 80)
logger.info("PIPELINE COMPLETED")
logger.info("=" * 80)
logger.info(f"Observation source: {obs_source_name}")
logger.info(f"Output directory: {output_dir}")
logger.info(f"\nGenerated/cached files:")

if use_cached_healda:
    logger.info(f"  - healda_analysis.nc (cached)")
elif not args.skip_healda:
    logger.info(f"  - healda_analysis.nc (newly generated)")
    obs_prefix = "merra21c" if args.use_merra21c else "ufs"
    logger.info(f"  - observations/{obs_prefix}_conv_obs_*.pkl")
    logger.info(f"  - observations/{obs_prefix}_sat_obs_*.pkl")
else:
    logger.info(f"  - era5_initial_conditions.nc")

if not args.skip_plots:
    logger.info(f"  - initial_conditions.png")

if args.models:
    for model_key in forecasts.keys():
        logger.info(f"  - {model_key}_forecast.nc")

    if not args.skip_plots and len(forecasts) > 0:
        logger.info(f"  - forecast_comparison_*.png")

    if 'metrics' in locals() and metrics:
        logger.info(f"  - forecast_metrics.json")
        if not args.skip_plots:
            logger.info(f"  - rmse_evolution.png")
else:
    logger.info(f"  (no forecast models executed)")

if use_cached_era5:
    logger.info(f"  - era5_validation.nc (cached)")
elif not args.skip_era5 and era5_forecast is not None:
    logger.info(f"  - era5_validation.nc (newly generated)")

if errors:
    logger.info(f"\nErrors encountered ({len(errors)} model(s)):")
    for model_key, error in errors.items():
        logger.info(f"  ✗ {model_key}: {error[:100]}...")

if args.models:
    logger.info(f"\nSuccessful forecasts: {len(forecasts)}/{len(args.models)} models")

logger.info("\n" + "="*80)
logger.info("Pipeline complete!")
logger.info("="*80)
