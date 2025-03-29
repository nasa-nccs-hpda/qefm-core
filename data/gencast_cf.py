import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

input_dir = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/")
fmodel = "FMGenCast"
yyyy = "2024"
mm = "12"
dd = "01"
files = sorted(input_dir.glob(f"*{yyyy}-{mm}-??_*.nc"))
file = files[0]
MAPL_GRAV = 9.80665
FILL_VALUE = 9.969209968386869e+36
ds = xr.open_dataset(file)

## Coordinates
# Time
long_name = "time"
begin_date = f"{yyyy}{mm}{dd}"
begin_time = 120000
time_increment = 120000
units = f"hours since {yyyy}-{mm}-{dd} 12:00:00"
calendar = "proleptic_gregorian"


# change value of time
ds['time'] = ds['time'].values/np.timedelta64(1, 'h')
# add attributes
ds.time.attrs = {
    "units" : units,
    "calendar" : calendar,
    "begin_date" : begin_date,
    "begin_time" : begin_time,
    "time_increment": time_increment
}

# Latitude
lats = ds['latitude'].values
fill_north = False
fill_south = False
if lats[0] > lats[-1]:
    # Flip the latitude array
    ds['lat'] = lats[::-1]
    # Flip the data array
    ds = ds.sel(lat=slice(None, None, -1))
if -90. not in ds.lat.values:
    fill_south = True
    lat_values = np.insert(ds.lat.values, 0, -90)
    ds = ds.assign_coords(lat=lat_values)
if 90. not in ds.lat.values:
    fill_north = True
    lat_values = np.append(ds.lat.values, 90)
    ds = ds.assign_coords(lat=lat_values)

ds.lat.attrs = {
    "long_name" : "latitude",
    "units" : "degrees_north",
}

# Longitude
lons = ds['lon'].values
if min(lons) == 0:
    ds['lon'] = ((ds["lon"] + 180) % 360) - 180
    ds = ds.sortby(ds.lon)
ds.lon.attrs = {
    "long_name" : "longitude",
    "units" : "degrees_east",
}

# level
ds = ds.rename({'level': 'lev'})
levs = ds['lev'].values
if levs[0] < levs[-1]:
    # Flip the level array
    ds['lev'] = levs[::-1]
    # Flip the data array
    ds = ds.sel(lev=slice(None, None, -1))
ds.level.attrs = {
    "long_name" : "pressure_level",
    "units" : "hPa",
}

# ensemble
ds = ds.rename({'sample': 'ens'})
ds.ens.attrs = {
    "long_name" : "ensemble_member",
    "units" : " ",
}

## Variables
# rename variables
rename_dict = {
    "10m_u_component_of_wind": "U10",
    "10m_v_component_of_wind": "V10",
    "2m_temperature": "T2M",
    "geopotential": "H",
    "mean_sea_level_pressure": "SLP",
    "sea surface temperature": "SST",
    "specific_humidity": "QV",
    "temperature": "T",
    "total_precipitation": "PRECTOT",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "vertical_velocity": "OMEGA",
}
ds = ds.rename(rename_dict)

# add geopotential at surface
source = Path("/discover/nobackup/projects/QEFM/data/FMGenCast")
tmp_file = list(source.glob(f"*{yyyy}-{mm}-{dd}_*.nc"))[0]
ds_temp = xr.open_dataset(files[0])
ds['PHIS'] = ds_temp['geopotential_at_surface']

# add attributes
varMap = {
    "U10": {
        "long_name" : "10-meter_eastward_wind",
        "units" : "m s-1",
    },
    "V10": {
        "long_name" : "10-meter_northward_wind",
        "units" : "m s-1",
    },
    "T2M": {
        "long_name" : "2-meter_air_temperature",
        "units" : "K",
    },
    "H": {
        "long_name" : "height",
        "units" : "m",
    },
    "SLP": {
        "long_name" : "sea_level_pressure",
        "units" : "Pa",
    },
    "SST": {
        "long_name" : "sea_surface_temperature",
        "units" : "K",
    },
    "QV": {
        "long_name" : "specific_humidity",
        "units" : "kg kg-1",
    },
    "T": {
        "long_name" : "air_temperature",
        "units" : "K",
    },
    "PRECTOT": {
        "long_name" : "total_precipitation",
        "units" : "m",
    },
    "U": {
        "long_name" : "eastward_wind",
        "units" : "m s-1",
    },
    "V": {
        "long_name" : "northward_wind",
        "units" : "m s-1",
    },
    "OMEGA": {
        "long_name" : "vertical_pressure_velocity",
        "units" : "Pa s-1",
    },
    "PHIS": {
        "long_name" : "surface_geopotential_height",
        "units" : "m+2 s-2",
    },
}
# add attributes
for var in ds.data_vars:
    ds[var].attrs = varMap[var]

## Mask variables based on elevation
topo = ds.PHIS.values/MAPL_GRAV
height = ds.H.values
mask = np.where(height > topo, 1, 0)
variables_3d = [var for var in ds.data_vars if 'lev' in ds[var].dims]
for var in variables_3d:
    ds[var] = ds[var].where(mask == 1, FILL_VALUE)

## add global attributes
ds.attrs = {
    "title" : f"{fmodel} {yyyy}-{mm}-{dd}",
    "institution" : "NASA CISTO Data Science Group",
    "source" : f"{fmodel} model output",
    "history" : f"created by data/geos_cf.py",
    "references" : "https://gmao.gsfc.nasa.gov",
    "Conventions" : "CF",
    "Comment" : "NetCDF-4" 
}


