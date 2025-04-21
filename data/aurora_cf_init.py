import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Convert Aurora init to CF-compliant NetCDF")
# parser.add_argument("input_dir", type=str, help="Path to GenCast output directory")
# parser.add_argument("fmodel", type=str, help="Model name")
parser.add_argument("--year", "-y", type=str, help="Year")
parser.add_argument("--month", "-m", type=str, help="Month")
parser.add_argument("--day", "-d", type=str, help="Day")
args = parser.parse_args()

#input_dir = Path("/discover/nobackup/projects/QEFM/data/")
input_dir = Path("/css/era5")
fmodel = "FMPangu"
yyyy = args.year
mm = args.month
dd = args.day

files = []
file_path = input_dir / "pressure_hourly" / "inst" / f"Y{yyyy}" / f"M{mm}" 
files.append(sorted(file_path.glob(f"era5_atmos-inst_allvar_{yyyy}{mm}{dd}_00z.nc"))[0])
file_path = input_dir / "surface_hourly" / "inst" / f"Y{yyyy}" / f"M{mm}" 
files.append(sorted(file_path.glob(f"era5_surface-inst_allvar_{yyyy}{mm}{dd}_00z.nc"))[0])



#file = files[0]
if len(files) != 2:
    raise ValueError("There should be 2 files for each day")
else:
    vars = ['z', 'q', 't', 'u', 'v']
    levs = levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    ds1 = xr.open_dataset(files[0])
    ds1 = ds1[vars].sel(pressure_level=levs)
    print(ds1)
    vars = ['msl', 'u10', 'v10', 't2m']
    ds2 = xr.open_dataset(files[1])
    ds2 = ds2[vars]
    print(ds2)
    ds = xr.merge([ds1, ds2], compat='override')

ds = ds.rename({'valid_time': 'time', 'pressure_level': 'lev', 'latitude': 'lat', 'longitude': 'lon'})
ds = ds.isel(time=0).expand_dims('time')

MAPL_GRAV = np.float32(9.80665)
FILL_VALUE = np.float32(1.e+20)
dt = pd.to_datetime(f"{yyyy}-{mm}-{dd} 00:00:00")

print("Processing file : ", files)
print("At Open : \n", ds)

## Add variable geopotential at surface
## will be used to mask variables based on elevation
source = Path("/discover/nobackup/projects/QEFM/data/FMAurora")
tmp_file = source / "static.nc"
ds_temp = xr.open_dataset(tmp_file)
pv = ds_temp['z'].squeeze().to_numpy()
ds['PHIS'] = (['lat', 'lon'], pv)

ds['LSM'] = (['lat', 'lon'], ds_temp['lsm'].squeeze().to_numpy())
ds['LSM'].attrs = {
    "var_name" : "LSM",
    "long_name" : "Land-sea mask",
    "units" : "1",
    "_FillValue" : FILL_VALUE,
}

ds['SLT'] = (['lat', 'lon'], ds_temp['slt'].squeeze().to_numpy())
ds['SLT'].attrs = {
    "var_name" : "SLT",
    "long_name" : "Soil type",
    "units" : "(Code table 4.213)",
    "_FillValue" : FILL_VALUE,
}
## Coordinates
# Time
HH = dt.strftime("%H")
YYYY = dt.strftime("%Y")
MM = dt.strftime("%m")
DD = dt.strftime("%d")
long_name = "time"
begin_date = np.int32(f"{YYYY}{MM}{DD}")
begin_time = np.int32(dt.hour*10000)
time_increment = np.int32(60000)

units = f"hours since {YYYY}-{MM}-{DD} {HH}:00:00"
calendar = "proleptic_gregorian"

# get time stamp for output file
#t = ds['time'].values
#tstamp = dt.strftime("%Y-%m-%dT%H") 
tstamp = dt.strftime("%Y%m%d_%Hz")
print("Time stamp : ", tstamp)


# change value of time
#ref_time = np.datetime64("2024-12-01T00:00:00", "ns")
#ds['time'] = np.float32((ds['time']-ref_time)/np.timedelta64(1, 'h'))
ds['time'] = np.float32((ds['time']-ds['time'])/np.timedelta64(1, 'h'))
#ds['time'].values = np.float32(0.0)
# add attributes
ds.time.attrs = {
    "long_name" : long_name,
    "units" : units,
    "calendar" : calendar,
    "begin_date" : begin_date,
    "begin_time" : begin_time,
    "time_increment": time_increment
}

# Latitude
lats = ds['lat'].values.astype(np.float32)
ds['lat'] = lats
fill_north = False
fill_south = False

if lats[0] > lats[-1]:
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
lons = ds['lon'].values.astype(np.float32)
ds['lon'] = lons
if min(lons) == 0:
    ds['lon'] = ((ds["lon"] + 180) % 360) - 180
    ds = ds.sortby(ds.lon)
ds.lon.attrs = {
    "long_name" : "longitude",
    "units" : "degrees_east",
}

# level
#ds = ds.rename({'level': 'lev'})
levs = ds['lev'].values.astype(np.float32)
ds['lev'] = levs
if levs[0] < levs[-1]:
    # Flip the level array
    ds['lev'] = levs[::-1]
    # Flip the data array
    ds = ds.sel(lev=slice(None, None, -1))
ds.lev.attrs = {
    "long_name" : "pressure_level",
    "units" : "hPa",
}
print("After coord \n", ds)
## Variables
# rename variables
rename_dict = {
    "u10": "U10M",
    "v10": "V10M",
    "t2m": "T2M",
    "z": "H",
    "msl": "SLP",
    "q": "QV",
    "t": "T",
    "u": "U",
    "v": "V",
}
ds = ds.rename(rename_dict)
print("After rename \n ", ds)



# map attributes
varMap = {
    "U10M": {
        "long_name" : "10-meter_eastward_wind",
        "units" : "m s-1",
    },
    "V10M": {
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
    "QV": {
        "long_name" : "specific_humidity",
        "units" : "kg kg-1",
    },
    "T": {
        "long_name" : "air_temperature",
        "units" : "K",
    },
    "U": {
        "long_name" : "eastward_wind",
        "units" : "m s-1",
    },
    "V": {
        "long_name" : "northward_wind",
        "units" : "m s-1",
    },
    "PHIS": {
        "long_name" : "surface_geopotential_height",
        "units" : "m+2 s-2",
    },
}

# define chunk size for each variable

## TODO : Need to check surface geopotential height from ERA5
## Get geopotential height 
ds['H'] = ds['H']/MAPL_GRAV

## mask variables based on elevation
# topo
topo = ds.PHIS.values/MAPL_GRAV
height = ds.H.values
mask = np.where(height > topo, 1, 0)
for var in varMap.keys():
    # add attributes
    ds[var].attrs = varMap[var]
    ds[var].attrs['_FillValue'] = FILL_VALUE
    #ds[var].attrs['missing_value'] = FILL_VALUE
    #ds[var].attrs['fmissing_value'] = FILL_VALUE
    # mask 3d variables
    if 'lev' in ds[var].dims:
        ds[var] = ds[var].where(mask == 1, FILL_VALUE)
# chunk 
# nlats = len(ds.lat)
# nlons = len(ds.lon)
# chunks_size = {"ens": 1, "time": 1, "lev": 1, "lat": nlats, "lon": nlons}
# ds = ds.chunk(chunks_size)
print("After variable \n", ds)

ds = ds.drop_vars(['expver', 'number'])
## add global attributes
ds.attrs = {
    "title" : f"{fmodel} initial input at {YYYY}-{MM}-{DD}T{HH}:00:00", 
    "institution" : "NASA CISTO Data Science Group",
    #"source" : f"{fmodel} model output",
    "Conventions" : "CF",
    "Comment" : "NetCDF-4" 
}

## Write to NetCDF
compression = {"zlib": True, 
            "complevel": 1,
            "shuffle": True,}
encoding = {var: compression for var in ds.data_vars}
#output_dir = Path(f"/discover/nobackup/projects/QEFM/data/rollout_outputs/{fmodel}/Y{yyyy}/M{mm}/D{dd}")
output_dir = Path(f"/discover/nobackup/projects/QEFM/data/rollout_outputs/{fmodel}/test/Y{yyyy}/M{mm}/D{dd}")
output_dir.mkdir(parents=True, exist_ok=True)
fname = f"init_aurora_{tstamp}.nc"
output_file = output_dir / fname
ds.to_netcdf(output_file, encoding=encoding, engine="netcdf4")



print("Finished \n", ds)
