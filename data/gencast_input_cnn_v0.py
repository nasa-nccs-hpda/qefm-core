import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import datetime
from pathlib import Path
import argparse
def expand_dims(ds, steps):
    # Expand the time dimension of the dataset
    orig_time = ds.time.values
    extra_steps = steps-2

    ds_last = ds.isel(time=1)
    new_times = pd.date_range(start=orig_time[-1], periods=extra_steps+1, freq="12h")[1:]

    repeated = ds_last.expand_dims(time=range(extra_steps)).copy(deep=True)
    repeated['time'] = new_times
    ds_extended = xr.concat([ds, repeated], dim='time')
    return ds_extended

parser = argparse.ArgumentParser(description="Download GenCast input data")
parser.add_argument("--year", "-y", type=str, help="Year of the data")
parser.add_argument("--month", "-m", type=str, help="Month of the data")
parser.add_argument("--day", "-d", type=str, help="Day of the data")
parser.add_argument("--nsteps", "-n", default=22, type=str, help="Number of time steps")
parser.add_argument("--coarsen", "-c", default=True, type=bool, help="If True, coarsen the data to 1p0 degree")

args = parser.parse_args()
date_str = f"{args.year}-{args.month}-{args.day}"
#date_str = '2024-12-12'
nsteps = int(args.nsteps)

#nsteps = 22 
GRAV = 9.80665
start_time = f"{date_str}T00:00"
time_steps = pd.date_range(start=start_time, periods=nsteps, freq="12h")
#output_dir = Path("/discover/nobackup/projects/QEFM/data/FMGenCast/12hr/samples")
output_dir = Path("/discover/nobackup/projects/QEFM/data/FMGenCast/12hr/geos_cnn")


levs = np.array(
    [50,  100,  150,  200,  250,  \
     300,  400,  500,  600,  700, \
     850,  925,  1000])

static = ["land_sea_mask",
          "geopotential_at_surface",]

var_2d = ["2m_temperature",
          "sea_surface_temperature",
          "mean_sea_level_pressure",
          "10m_u_component_of_wind",
          "10m_v_component_of_wind",
          "total_precipitation",]

var_3d = ["temperature",
          "specific_humidity",
          "u_component_of_wind",
          "v_component_of_wind",
          "vertical_velocity",
          "geopotential",]
var_mapping = {
                "t2m_pred": "2m_temperature",
                "t_pred": "temperature",
                "u10_pred": "10m_u_component_of_wind",
                "v10_pred": "10m_v_component_of_wind",
                "u_pred": "u_component_of_wind",
                "v_pred": "v_component_of_wind",
                "q_pred": "specific_humidity",
                # "w": "vertical_velocity",
                "z_pred": "geopotential",
                "sst_pred": "sea_surface_temperature",
                "msl_pred": "mean_sea_level_pressure",
                "tp_pred": "total_precipitation_12hr",
                # "zs": "geopotential_at_surface",
                # "latitude": "lat",
                # "longitude": "lon",
                "lev": "level",
                }
               
var_list = var_mapping.keys()
nlev = len(levs)

# # get ear5 from gs
# ds = xr.open_dataset(
#     "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
#     engine="zarr",
#     #chunks={},
#     storage_options={"token": None}  # Public dataset, so no authentication needed
# )[var_list].sel(time=time_steps, level=levs)

# get dataset
#input_root = Path('/discover/nobackup/projects/QEFM/data/FMGenCast/geos-fp-interp')
input_path = Path('/discover/nobackup/khbreen/qefm_local/ML/lev_to_press/for_Gencast/best_output')
#input_path = input_root / f"Y{args.year}" / f"M{args.month}"
stamp = f"{args.year}{args.month}{args.day}"
#input_root = Path("/discover/nobackup/jli30/fromArlindo/output")
fs = list(input_path.glob(f"era5_pred_{stamp}_00z.nc"))+list(input_path.glob(f"era5_pred_{stamp}_12z.nc"))


#ds = xr.open_mfdataset(fs[:2], combine='nested', concat_dim='time', chunks=None)
ds = xr.concat([xr.open_dataset(fp) for fp in fs], dim='time')

# keep predicted variables
pred_vars = [var for var in ds.data_vars if var.endswith("_pred")]
ds = ds[pred_vars]

# subset the vertical levels
ds = ds.sel(lev=levs)


# ds_3d = xr.concat([xr.open_dataset(fp) for fp in fs[2:]], dim='time')
# ds = xr.merge([ds_2d, ds_3d])

# coarsen the data & reverse the latitude
# if args.coarsen:
ds = ds.isel(lat=slice(None, None, -4), lon=slice(None, None, 4)).compute()
res=1.0

ds = ds.drop_vars(["number", "expver"])
# change variable names
ds = ds.rename(var_mapping)

# compute mean across level for 2d variables
for var in var_2d:
    if var in ds.data_vars:
        ds[var] = ds[var].mean(dim="level", keep_attrs=True)
# # Geopotential from geopotential_height
# ds['geopotential'] = ds['geopotential'] * GRAV

# # rename the precipitation variable
# ds = ds.rename({
#     "total_precipitation": "total_precipitation_12hr",
# })
# change data types
ds = ds.astype({var: 'float32' for var in ds.data_vars})

# expand the time dimension
ds = expand_dims(ds, nsteps)

ds = ds.assign_coords(datetime=ds["time"])

# expand the dimensions 
for var in list(ds.data_vars) + ['datetime']:
    ds[var] = ds[var].expand_dims("batch")

# change time coordinate to timedelta
ds['time']=ds['time']-ds['time'].isel(time=0)
#ds['time']=ds['time']/np.timedelta64(1, 's')

# add land_sea_mask
file = f"/discover/nobackup/projects/QEFM/data/FMGenCast/12hr/Y2024/gencast-dataset-source-era5_date-{date_str}_res-1.0_levels-13_steps-20.nc"
ds_lsm = xr.open_dataset(file)
ds['land_sea_mask'] = ds_lsm['land_sea_mask']
ds['vertical_velocity'] = ds_lsm['vertical_velocity']
ds['geopotential_at_surface'] = ds_lsm['geopotential_at_surface']
# using the sea_surface_temperature from the original dataset
# ds['sea_surface_temperature'] = ds_lsm['sea_surface_temperature']

# drop the time dimension for land_sea_mask and geopotential_at_surface
#ds['land_sea_mask'] = ds['land_sea_mask'].isel(time=0).drop_vars("time")
#ds['geopotential_at_surface'] = ds['geopotential_at_surface'].isel(time=0).drop_vars(["time"])

# mask out skin temperature over land
#ds['sea_surface_temperature'] = ds['sea_surface_temperature'].where(ds['land_sea_mask'] == 0)

# writing to netcdf
output_file = output_dir / \
f"gencast-dataset-source-cnn\
_date-{date_str}_res-{str(res)}\
_levels-{str(nlev)}_steps-{str(nsteps-2)}.nc"
print(output_file)
ds.to_netcdf(output_file)


