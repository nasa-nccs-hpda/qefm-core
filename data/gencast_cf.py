import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

input_dir = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/")
fmodel = "FMGenCast"
file_path = input_dir / fmodel
yyyy = "2024"
mm = "12"
dd = "01"
files = sorted(file_path.glob(f"*{yyyy}-{mm}*_*.nc"))
file = files[0]

MAPL_GRAV = 9.80665
FILL_VALUE = 1.e+15
ds = xr.open_dataset(file)
print("At Open : \n", ds)

## Coordinates
# For GenCast Only, remove "batch"
ds = ds.squeeze(dim="batch")

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
lats = ds['lat'].values
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
levs = ds['lev'].values.astype(np.float32)
if levs[0] < levs[-1]:
    # Flip the level array
    ds['lev'] = levs[::-1]
    # Flip the data array
    ds = ds.sel(lev=slice(None, None, -1))
ds.lev.attrs = {
    "long_name" : "pressure_level",
    "units" : "hPa",
}

# ensemble
ds = ds.rename({'sample': 'ens'})
ds.ens.attrs = {
    "long_name" : "ensemble_member",
    "units" : " ",
}
print("After coord \n", ds)

## Variables
# rename variables
rename_dict = {
    "10m_u_component_of_wind": "U10",
    "10m_v_component_of_wind": "V10",
    "2m_temperature": "T2M",
    "geopotential": "H",
    "mean_sea_level_pressure": "SLP",
    "sea_surface_temperature": "SST",
    "specific_humidity": "QV",
    "temperature": "T",
    "total_precipitation_12hr": "PRECTOT",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "vertical_velocity": "OMEGA",
}
ds = ds.rename(rename_dict)
print("After rename \n ", ds)

# add variable geopotential at surface
source = Path("/discover/nobackup/projects/QEFM/data/FMGenCast")
tmp_file = list(source.glob(f"*{yyyy}-{mm}-{dd}_*.nc"))[0]
ds_temp = xr.open_dataset(tmp_file)
ds['PHIS'] = ds_temp['geopotential_at_surface']

# map attributes
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

# define chunk size for each variable
# mask variables based on elevation
## TODO : Need to check surface geopotential height from ERA5

# topo
topo = ds.PHIS.values/MAPL_GRAV
height = ds.H.values
mask = np.where(height > topo, 1, 0)
# chunk
nlats = len(ds.lat)
nlons = len(ds.lon)
chunks_2d = {"ens": 1, "time": 1, "lat": nlats, "lon": nlons}
chunks_3d = {"ens": 1, "time": 1, "lev": 1, "lat": nlats, "lon": nlons}
chunk_sizes = {}
for var in ds.data_vars:
    # add attributes
    ds[var].attrs = varMap[var]
    # set chunk size
    if 'lev' in ds[var].dims:
        # also mask 3D variables
        ds[var] = ds[var].where(mask == 1, FILL_VALUE)
        chunk_sizes[var] = chunks_3d
    else:
        chunk_sizes[var] = chunks_2d 
ds = ds.chunk(chunk_sizes)
print("After variable \n", ds)

## add global attributes
ds.attrs = {
    "title" : f"{fmodel} forecast start at {yyyy}-{mm}-{dd}T12:00:00", 
    "institution" : "NASA CISTO Data Science Group",
    "source" : f"{fmodel} model output",
    "Conventions" : "CF",
    "Comment" : "NetCDF-4" 
}

## Write to NetCDF
compression = {"zlib": True, 
               "complevel": 1,
               "shuffle": True,}
encoding = {var: compression for var in ds.data_vars}
output_dir = Path(f"/discover/nobackup/projects/QEFM/data/rollout_outputs/{fmodel}/CF")
output_dir.mkdir(parents=True, exist_ok=True)
fname = f"{fmodel}-prediction-era5_date-{yyyy}-{mm}-{dd}_res-1.0_levels-13_steps-20.nc"
output_file = output_dir / fname
ds.to_netcdf(output_file, encoding=encoding, engine="netcdf4")



print("Finished \n", ds)
