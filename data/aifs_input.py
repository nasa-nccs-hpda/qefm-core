import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import datetime
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Download GenCast input data")
parser.add_argument("--year", "-y", type=str, help="Year of the data")
parser.add_argument("--month", "-m", type=str, help="Month of the data")
parser.add_argument("--day", "-d", type=str, help="Day of the data")
parser.add_argument("--nsteps", "-n", default=2, type=str, help="Number of time steps")
parser.add_argument("--coarsen", "-c", default=False, type=bool, help="If True, coarsen the data to 1p0 degree")

args = parser.parse_args()
date_str = f"{args.year}-{args.month}-{args.day}"
nsteps = int(args.nsteps) 
end_time = f"{date_str}T00:00"
time_steps = pd.date_range(end=end_time, periods=nsteps, freq="6h")
output_dir = Path("/discover/nobackup/projects/QEFM/data/FMAifs/nc_files")

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
SFC_LONG_NAME = ["10m_u_component_of_wind",
                 "10m_v_component_of_wind",
                 "2m_dewpoint_temperature",
                 "2m_temperature",
                 "mean_sea_level_pressure",
                 "skin_temperature",
                 "surface_pressure",
                 "total_column_water",
                 "land_sea_mask",
                 "geopotential_at_surface",
                 "slope_of_sub_gridscale_orography",
                 "standard_deviation_of_orography"]
PARAM_PL = ["z", "t", "u", "v", "w", "q"]
PL_LONG_NAME = ["geopotential",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
                "specific_humidity"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# levs = np.array(
#     [50,  100,  150,  200,  250,  \
#      300,  400,  500,  600,  700, \
#      850,  925,  1000])

# static = ["land_sea_mask",
#           "geopotential_at_surface",]

# var_2d = ["10m_u_component_of_wind",
#           "10m_v_component_of_wind",
#           "2m_dewpoint_temperature",
#           "2m_temperature",
#           "mean_sea_level_pressure",
#           "skin_temperature",
#           "surface_pressure",
#           "total_column_water",
#           "land_sea_mask",
#           "geopotential_at_surface",
#           "slope_of_sub_gridscale_orography",
#           "standard_deviation_of_orography",
#           ]

# var_3d = [ "geopotential",
#           "temperature",
#           "u_component_of_wind",
#           "v_component_of_wind",
#           "vertical_velocity",
#           "specific_humidity",
#           ]

var_list = SFC_LONG_NAME + PL_LONG_NAME
res = 0.25
nlev = len(LEVELS)

# get ear5 from gs
ds = xr.open_dataset(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    engine="zarr",
    #chunks={},
    storage_options={"token": None}  # Public dataset, so no authentication needed
)[var_list].sel(time=time_steps, level=LEVELS)

# coarsen the data & reverse the latitude
if args.coarsen:
    ds = ds.isel(latitude=slice(None, None, -4), longitude=slice(None, None, 4))
    res = 1.0

# # change dimension names
# ds = ds.rename({
#     "latitude": "lat",
#     "longitude": "lon",
# })

ds = ds.assign_coords(datetime=ds["time"])

# change time coordinate to timedelta
ds['time']=ds['time']-ds['time'].isel(time=0)
#ds['time']=ds['time']/np.timedelta64(1, 's')

# # drop the time dimension for land_sea_mask and geopotential_at_surface
# ds['land_sea_mask'] = ds['land_sea_mask'].isel(time=0).drop_vars("time")
# ds['geopotential_at_surface'] = ds['geopotential_at_surface'].isel(time=0).drop_vars("time")

# # expand the dimensions 
# for var in var_2d + var_3d + ['datetime']:
#     ds[var] = ds[var].expand_dims("batch")

# # rename the precipitation variable
# ds = ds.rename({
#     "total_precipitation": "total_precipitation_12hr",
# })
# writing to netcdf
output_file = output_dir / \
f"aifs-dataset-source-era5\
_date-{date_str}_res-{str(res)}\
_levels-{str(nlev)}_steps-{str(nsteps-2)}.nc"
print(output_file)
ds.to_netcdf(output_file)


