import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

print("Download and subset ERA5 input data:")
parser = argparse.ArgumentParser(description="Download and subset ERA5 input data:")
parser.add_argument("--outdir", "-o", default="/discover/nobackup/projects/QEFM/data", type=str, help="Output directory")
parser.add_argument("--year", "-y", default="24", type=str, help="Year of the data")
parser.add_argument("--month", "-m", default="12", type=str, help="Month of the data")
parser.add_argument("--day", "-d", default="01", type=str, help="Day of the data")
parser.add_argument("--freq", "-f", default="12h", type=str, help="Frequency in hours")
parser.add_argument("--nsteps", "-n", default=22, type=str, help="Number of time steps")
parser.add_argument("--coarsen", "-c", default=False, type=bool, help="If True, coarsen the data to 1p0 degree")

args = parser.parse_args()
date_str = f"{args.year}-{args.month}-{args.day}"
nsteps = int(args.nsteps) 
start_time = f"{date_str}T00:00"
cfreq=f"{args.freq}"
output_dir=Path(f"{args.outdir}")
print("arguments:", args._get_kwargs)
time_steps = pd.date_range(start=start_time, periods=nsteps, freq=cfreq)

levs = np.array(
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, \
    250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, \
    850, 875, 900, 925, 950, 975, 1000])

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

var_list = static + var_2d + var_3d
res = 0.25
nlev = len(levs)

# get ear5 from gs
ds = xr.open_dataset(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    engine="zarr",
    #chunks={},
    storage_options={"token": None}  # Public dataset, so no authentication needed
)[var_list].sel(time=time_steps, level=levs).isel(latitude=slice(None, None, -1))

# coarsen the data & reverse the latitude
if args.coarsen:
    ds = ds.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
    res = 1.0

# change dimension names
ds = ds.rename({
    "latitude": "lat",
    "longitude": "lon",
})

ds = ds.assign_coords(datetime=ds["time"])

# change time coordinate to timedelta
ds['time']=ds['time']-ds['time'].isel(time=0)
#ds['time']=ds['time']/np.timedelta64(1, 's')

# drop the time dimension for land_sea_mask and geopotential_at_surface
ds['land_sea_mask'] = ds['land_sea_mask'].isel(time=0).drop_vars("time")
ds['geopotential_at_surface'] = ds['geopotential_at_surface'].isel(time=0).drop_vars("time")

# expand the dimensions 
for var in var_2d + var_3d + ['datetime']:
    ds[var] = ds[var].expand_dims("batch")

# rename the precipitation variable
ds = ds.rename({
    "total_precipitation": "total_precipitation_6hr",
})
# writing to netcdf
output_file = output_dir / \
f"graphcast-dataset-source-era5\
_date-{date_str}_res-{str(res)}\
_levels-{str(nlev)}_freq-{str(cfreq)}_steps-{str(nsteps-2)}.nc"
print(output_file)
ds.to_netcdf(output_file)
