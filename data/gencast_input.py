import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import datetime

date_str = "2019-03-29"

start_time = f"{date_str}T00:00"
time_steps = pd.date_range(start=start_time, periods=3, freq="12h")

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

var_list = static + var_2d + var_3d

# get ear5 from gs
ds = xr.open_dataset(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    engine="zarr",
    #chunks={},
    storage_options={"token": None}  # Public dataset, so no authentication needed
)[var_list].sel(time=time_steps, level=levs)

# coarsen the data & reverse the latitude
ds = ds.isel(latitude=slice(None, None, -4), longitude=slice(None, None, 4))

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
    "total_precipitation": "total_precipitation_12hr",
})      
# writing to netcdf
ds.to_netcdf(f"gencast-dataset-source-era5_date-{date_str}_res-1.0_levels-13_steps-01.nc")


