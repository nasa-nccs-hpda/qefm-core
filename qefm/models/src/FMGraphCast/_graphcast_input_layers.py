import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import glob
import warnings
import os

print("Download and subset ERA5 input data:")
parser = argparse.ArgumentParser(description="Download and subset ERA5 input data:")
parser.add_argument("--outdir", "-o", default="/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/6h/Y2024/var/v20240902_1437", type=str, help="Output directory")
parser.add_argument("--year", "-y", default="2024", type=str, help="Year of the data")
parser.add_argument("--month", "-m", default="12", type=str, help="Month of the data")
parser.add_argument("--day", "-d", default="01", type=str, help="Day of the data")
parser.add_argument("--freq", "-f", default="6h", type=str, help="Frequency in hours")
parser.add_argument("--levs", "-l", default="37", type=str, help="Number of pressure levels")
parser.add_argument("--nsteps", "-n", default=42, type=str, help="Number of time steps")
parser.add_argument("--var", "-v", default="None", type=str, help="Parameter of the data")
parser.add_argument("--coarsen", "-c", default=False, type=bool, help="If True, coarsen the data to 1p0 degree")

args = parser.parse_args()
date_str = f"{args.year}-{args.month}-{args.day}"
nsteps = int(args.nsteps) 
nlevs = int(args.levs) 
start_time = f"{date_str}T00:00"
cfreq=f"{args.freq}"
var=f"{args.var}"
output_dir=Path(f"{args.outdir}")
print("arguments:", args._get_kwargs)
time_steps = pd.date_range(start=start_time, periods=nsteps, freq=cfreq)

##################
levs3 = np.array(
    [1, 500, 825])

levs30 = np.array(
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, \
    250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825])

levs37 = np.array(
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, \
    250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, \
    850, 875, 900, 925, 950, 975, 1000])
##################

if (nlevs==37):
    levs=levs37
elif (nlevs==30):
    levs=levs30
else:
    levs=levs3
#print(levs)

# static = ["geopotential_at_surface",]

# var_2d = ["2m_temperature",]

# var_3d = ["geopotential",]

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

if (var != "None"):
    var_3d=[var]

res = 0.25
nlev = len(levs)
var_list = static + var_2d
print("Subset 2D vars: ", var_list)

for var in var_list:
    print("download 2D var:", var)
    filename=f"{args.outdir}/graphcast-dataset-source-era5_date-{date_str}_var-{var}_res-{str(res)}_levels-{str(nlev)}_freq-{str(cfreq)}_steps-{str(nsteps)}.nc"
    if os.path.exists(Path(filename)):
        print("Source file already exists: ", filename)
    else:
        print("Source file doesn't exist: ", filename)

        # get ear5 from gs
        ds = xr.open_dataset(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            engine="zarr",
            #chunks={},
            storage_options={"token": None}  # Public dataset, so no authentication needed
        )[var].sel(time=time_steps).isel(latitude=slice(None, None, -1))

        # coarsen the data & reverse the latitude
        if args.coarsen:
            ds = ds.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
            res = 1.0

        # change dimension names
        ds = ds.rename({
            "latitude": "lat",
            "longitude": "lon",
        })

        # change time coordinate to timedelta
#        if var in (var_2d):
        ds = ds.assign_coords(datetime=ds["time"])
        
        # change time coordinate to timedelta
#        print(str(ds['time']))
        ds['time']=ds['time']-ds['time'].isel(time=0)
#        ds.time.encoding["units"] = "hours since 1900-01-01"
#        print(str(ds['time']))
    #    ds['datetime'] = ds['datetime'].expand_dims("batch")
#        print(ds.datetime)
        ds.datetime.expand_dims("batch")
#        print(ds.datetime)

        # if var in (var_2d + var_3d):
        #     if var in ds:
        #         ds[var] = ds[var].expand_dims("batch")
        #         print('expanded 2d dims early: ', var)    

        # writing to netcdf
        output_file = output_dir / \
        f"graphcast-dataset-source-era5_date-{date_str}_var-{var}_res-{str(res)}_levels-{str(nlev)}_freq-{str(cfreq)}_steps-{str(nsteps)}.nc"
        print(output_file)
        ds.to_netcdf(output_file)

var_list = var_3d
print("Subset 3D vars: ", var_list)
for var in var_list:
    print("download 3D var:", var)
    filename=f"{args.outdir}/graphcast-dataset-source-era5_date-{date_str}_var-{var}_res-{str(res)}_levels-{str(nlev)}_freq-{str(cfreq)}_steps-{str(nsteps)}.nc"
    if os.path.exists(Path(filename)):
        print("Source file already exists: ", filename)
    else:
        print("Source file doesn't exist: ", filename)

        # get ear5 from gs
        ds = xr.open_dataset(
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            engine="zarr",
            #chunks={},
            storage_options={"token": None}  # Public dataset, so no authentication needed
        )[var].sel(time=time_steps, level=levs).isel(latitude=slice(None, None, -1))

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
#        ds.time.encoding["units"] = "hours since 1900-01-01"
        #ds['datetime'] = ds['datetime'].expand_dims("batch")
#        print(ds.datetime)
        ds.datetime.expand_dims("batch")
#        print(ds.datetime)

        # if var in (var_2d + var_3d):
        #     if var in ds:
        #         ds[var] = ds[var].expand_dims("batch")
        #         print('expanded 3d dims early: ', var)    

        # writing to netcdf
        output_file = output_dir / \
        f"graphcast-dataset-source-era5_date-{date_str}_var-{var}_res-{str(res)}_levels-{str(nlev)}_freq-{str(cfreq)}_steps-{str(nsteps)}.nc"
        print(output_file)
        ds.to_netcdf(output_file)

# Define the path to your NetCDF files, using a wildcard for multiple files
# This assumes your files are named something like 'data_2000.nc', 'data_2001.nc', etc.
# and contain 'time', 'level', 'lat', 'lon' dimensions.
source = f"{args.outdir}/gr*{date_str}*.nc"
file_paths = glob.glob(source)
print(file_paths)

def _preprocess_func(ds):
    print("in stub preprocess_func", str(ds))

def preprocess_func(_ds):
    # Example: Rename a variable and add a new coordinate

    # if 'time' in ds:
    #     ds = ds.assign_coords(datetime=ds["time"])
    #     ds['time']=ds['time']-ds['time'].isel(time=0)
    # if 'datetime' in ds:
    #     ds['datetime'] = ds['datetime'].expand_dims("batch")

    # ds = _ds.drop_vars("datetime")
    # # set/override datetime attributes 
    # ds['datetime'].attrs['units'] = 'hours since 1900-01-01'
    # ds['datetime'].attrs['calendar'] = 'proleptic_gregorian'

    # print(ds['datetime'].values)
    # ds['datetime'] = ds['datetime'].to_pandas()
    # print(ds['datetime'].values)
    # reference_date = pd.Timestamp('1900-01-01')
    # time_delta = ds['datetime'] - reference_date
    # hours_since_1900 = time_delta.dt.total_seconds() / 3600
    # ds['time'] = hours_since_1900
    # # Optionally, update attributes for clarity
    # ds['time'].attrs['units'] = 'hours since 1900-01-01'
    ds=_ds
#    print(str(ds))
    if "total_precipitation" in ds:
        print("renaming total_precipitation to total_precipitation_6hr")
        ds["total_precipitation"] = ds["total_precipitation"].expand_dims("batch")
        ds = ds.rename({"total_precipitation": "total_precipitation_6hr"})

    if 'land_sea_mask' in ds:
        ds['land_sea_mask'] = ds['land_sea_mask'].isel(time=0).drop_vars("time")
#        del ds["land_sea_mask"].attrs["coordinates"]
        print('dropped time var from geopotential_at_surface')
    if 'geopotential_at_surface' in ds:
        ds['geopotential_at_surface'] = ds['geopotential_at_surface'].isel(time=0).drop_vars("time")
        print(ds["geopotential_at_surface"].attrs)
        #del ds["geopotential_at_surface"].attrs["coordinates"]
        print('dropped time var from geopotential_at_surface')

    # if 'time' in ds:
    #     print('found time: ', str(ds['time']))   

    if 'datetime' in ds:
#        print('expanding datetime: ', str(ds['datetime']))   
        ds['datetime'] = ds['datetime'].expand_dims("batch")
#        print('expanded datetime: ', str(ds['datetime']))    

    for var in var_2d + var_3d:
        if var in ds:
            ds[var] = ds[var].expand_dims("batch")
#            print('expanded dims for: ', var)   


    # # To get hours since 1900-01-01 as a numerical value:
    # reference_date = np.datetime64('1900-01-01T00:00:00')
    # hours_since_ref = (ds['time'].values - reference_date) / np.timedelta64(1, 'h')
    # print(hours_since_ref)
    
    # reference_date = np.datetime64('2024-12-01T00:00:00')
    # hours_since_ref = (ds['time'].values - reference_date) / np.timedelta64(1, 'h')
    # print(hours_since_ref)

    # print(ds['datetime'])
    # ds['datetime'].encoding['units'] = "hours since 1900-01-01"
    # print(ds['datetime'])
    
    # set/override global attributes 
    ds.attrs["valid_time_start"] = "1940-01-01"
    ds.attrs["last_updated"] = "2025-07-29 01:45:36.622817+00:00"    
    ds.attrs["valid_time_stop"] = "2025-04-30"
    ds.attrs["valid_time_stop_era5t"] = "2025-07-23"

    return ds

ds = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    # Open the multiple files as a single dataset
    # concat_dim='time' tells xarray to concatenate along the 'time' dimension
    # parallel=True enables parallel processing using dask for potentially faster loading
    # chunks can be specified to control how data is loaded into memory (e.g., for dask)
    print(file_paths)
    ds = xr.open_mfdataset(
        file_paths,
        #concat_dim='time',
        combine='by_coords',
        #combine='nested',  # Use 'nested' for more explicit control over concatenation
        parallel=False,
        #parallel=True,
        preprocess=preprocess_func,
        chunks={'time': 10, 'level': 37, 'lat': 721, 'lon': 1440}, # Example chunking
    #    chunks={'time': 10, 'level': 5, 'lat': 100, 'lon': 100} # Example chunking
        decode_times=False
    )

# Now you can work with the combined dataset
print(ds)
# writing to netcdf
output_file = output_dir / \
f"aggregated_graphcast-dataset-source-era5_date-{date_str}_var-ALL_res-{str(res)}_levels-{str(nlev)}_freq-{str(cfreq)}_steps-{str(nsteps)}.nc"
ds.to_netcdf(output_file)