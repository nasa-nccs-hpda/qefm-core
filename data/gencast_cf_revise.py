import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def get_phis(ds, fmodel, yyyy, mm, dd):
    """
    Get surface geopotential height from input file
    """
    source = Path("/discover/nobackup/projects/QEFM/data")
    file_path = source / fmodel / '0p25'
    tmp_file = sorted(file_path.glob(f"*{yyyy}-{mm}-{dd}*.nc"))
    print("INside ", tmp_file)
    ds_temp = xr.open_dataset(tmp_file[0])
    ds['PHIS'] = ds_temp['geopotential_at_surface']
    return ds

def update_time(ds, ref_date, ctime):
    """
    Update time coordinate in the dataset
    """
    dt = pd.to_datetime(ref_date + ctime)
    HH = dt.strftime("%H")
    YYYY = dt.strftime("%Y")
    MM = dt.strftime("%m")
    DD = dt.strftime("%d")
    tstamp = dt.strftime("%Y-%m-%dT%H") 

    # Time
    long_name = "time"
    begin_date = np.int32(f"{YYYY}{MM}{DD}")
    begin_time = np.int32(dt.hour * 10000)
    time_increment = np.int32(120000)
    units = f"hours since {YYYY}-{MM}-{DD} {HH}:00:00"
    calendar = "proleptic_gregorian"

    # change value of time
    ds['time'] = np.float32((ds['time']- ds['time'])/np.timedelta64(1, 'h'))
    # add attributes
    ds.time.attrs = {
        "long_name" : long_name,
        "units" : units,
        "calendar" : calendar,
        "begin_date" : begin_date,
        "begin_time" : begin_time,
        "time_increment": time_increment
    }
    return ds, tstamp

def update_lat(ds):
    """
    Update latitude coordinate in the dataset
    """
    lats = ds['lat'].values
    if lats[0] > lats[-1]:
        ds = ds.sel(lat=slice(None, None, -1))
    ds.lat.attrs = {
        "long_name" : "latitude",
        "units" : "degrees_north",
    }
    return ds

def update_lon(ds):
    """
    Update longitude coordinate in the dataset
    """
    lons = ds['lon'].values
    if min(lons) == 0:
        ds['lon'] = ((ds["lon"] + 180) % 360) - 180
        ds = ds.sortby(ds.lon)
    ds.lon.attrs = {
        "long_name" : "longitude",
        "units" : "degrees_east",
    }
    return ds

def update_level(ds):
    """
    Update level coordinate in the dataset
    """
    ds = ds.rename({'level': 'lev'})
    levs = ds['lev'].values.astype(np.float32)
    if levs[0] < levs[-1]:
        ds = ds.sel(lev=slice(None, None, -1))
    ds.lev.attrs = {
        "long_name" : "pressure_level",
        "units" : "hPa",
    }
    return ds

def rename_variables(ds):
    """
    Rename variables in the dataset to CF-compliant names &
    add attributes
    """
    rename_dict = {
        "10m_u_component_of_wind": "U10M",
        "10m_v_component_of_wind": "V10M",
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
    # rename variables
    ds = ds.rename(rename_dict)
    # add attributes
    for var in ds.data_vars:
        ds[var].attrs = varMap[var]
        ds[var].attrs['_FillValue'] = np.float32(1.e+15)
        ds[var].attrs['missing_value'] = np.float32(1.e+15)
        ds[var].attrs['fmissing_value'] = np.float32(1.e+15)
    return ds

def mask_variables(ds, MAPL_GRAV):
    """
    Mask variables based on elevation
    """
    # Get geopotential height
    ds['H'] = ds['H']/MAPL_GRAV
    # Topography
    topo = ds.PHIS.values/MAPL_GRAV
    height = ds.H.values
    mask = np.where(height > topo, 1, 0)
    for var in ds.data_vars:
        if 'lev' in ds[var].dims:
            ds[var] = ds[var].where(mask == 1, np.float32(1.e+15))
    return ds

def add_global_attributes(ds, fmodel, yyyy, mm, dd):
    """
    Add global attributes to the dataset
    """
    ds.attrs = {
        "title" : f"{fmodel} forecast start at {yyyy}-{mm}-{dd}T12:00:00", 
        "institution" : "NASA CISTO Data Science Group",
        "source" : f"{fmodel} model output",
        "Conventions" : "CF",
        "Comment" : "NetCDF-4" 
    }
    return ds

def write_to_netcdf(ds, output_dir, fmodel, tstamp):
    """
    Write the dataset to NetCDF file
    """
    compression = {"zlib": True, 
                "complevel": 1,
                "shuffle": True,}
    encoding = {var: compression for var in ds.data_vars}
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{fmodel}-prediction-era5_date-{tstamp}_res-1.0_levels-13_ens-mean.nc"
    output_file = output_dir / fname
    ds.to_netcdf(output_file, encoding=encoding, engine="netcdf4")

def proc_slice(ds, ref_date, ctime):
    """
    Process a slice of the dataset
    """
    # Time coordinate
    ds, tstamp = update_time(ds, ref_date, ctime)
    # Latitude coordinate
    ds = update_lat(ds)
    # Longitude coordinate
    ds = update_lon(ds)
    # Level coordinate
    ds = update_level(ds)
    # Rename variables
    ds = rename_variables(ds)
    # Mask variables based on elevation
    ds = mask_variables(ds, MAPL_GRAV)
    return ds, tstamp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert GenCast output to CF-compliant NetCDF")
    # parser.add_argument("input_dir", type=str, help="Path to GenCast output directory")
    # parser.add_argument("fmodel", type=str, help="Model name")
    parser.add_argument("--year", "-y", type=str, help="Year")
    parser.add_argument("--month", "-m", type=str, help="Month")
    parser.add_argument("--day", "-d", type=str, help="Day")
    args = parser.parse_args()

    # Parameters
    input_dir = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/")
    fmodel = "FMGenCast"
    file_path = input_dir / fmodel / 'raw' / '0p25'
    yyyy = args.year
    mm = args.month
    dd = args.day

    # Constants
    MAPL_GRAV = 9.80665
    FILL_VALUE = np.float32(1.e+15)
    ref_date = np.datetime64(f"{yyyy}-{mm}-{dd}T12:00:00")
    ens_num = 8

    # Read a file
    file = sorted(file_path.glob(f"*{yyyy}-{mm}-{dd}_*.nc"))
    ds_org = xr.open_dataset(file[0], engine='netcdf4')
    # For GenCast Only, remove "batch"
    ds_org = ds_org.squeeze(dim="batch")
    print(ds_org)

    # add variable geopotential at surface
    ds_org = get_phis(ds_org, fmodel, yyyy, mm, dd)

    for ctime in ds_org.time.values:
        ds = ds_org.sel(time=ctime).expand_dims("time")

        for idx in range(ens_num+1):
            print(idx)
            print(ds)
            print(ds.isel(sample=idx))
            if idx == 8:
                ds = ds.mean(dim='sample').squeeze()
            else:
                ds = ds.isel(sample=idx)
            
            ds = proc_slice(ds, ref_date, ctime)
            print(ds)

'''

        # Time coordinate
        ds, tstamp = update_time(ds, ref_date, ctime)
        # Latitude coordinate
        ds = update_lat(ds)
        # Longitude coordinate
        ds = update_lon(ds)
        # Level coordinate
        ds = update_level(ds)

        # Rename variables
        ds = rename_variables(ds)   





files = sorted(file_path.glob(f"*{yyyy}-{mm}-{dd}_*.nc"))
file = files[0]


ds_org = xr.open_dataset(file)
print("At Open : \n", ds_org)


## Coordinates
# For GenCast Only, remove "batch"
ds_org = ds_org.squeeze(dim="batch")

# add variable geopotential at surface
source = Path("/discover/nobackup/projects/QEFM/data/FMGenCast")
tmp_file = list(source.glob(f"*{yyyy}-{mm}-{dd}_*.nc"))[0]
ds_temp = xr.open_dataset(tmp_file)
ds_org['PHIS'] = ds_temp['geopotential_at_surface']
ens_mean = True

for ctime in ds_org.time.values:
    ds = ds_org.sel(time=ctime).expand_dims("time")
    dt = pd.to_datetime(ref_date + ctime)
    HH = dt.strftime("%H")
    YYYY = dt.strftime("%Y")
    MM = dt.strftime("%m")
    DD = dt.strftime("%d")

    tstamp = dt.strftime("%Y-%m-%dT%H") 

    # Time
    long_name = "time"
    begin_date = np.int32(f"{YYYY}{MM}{DD}")
    begin_time = np.int32(dt.hour * 10000)
    time_increment = np.int32(120000)
    units = f"hours since {YYYY}-{MM}-{DD} {HH}:00:00"
    calendar = "proleptic_gregorian"


    # change value of time
    ds['time'] = np.float32((ds['time']- ds['time'])/np.timedelta64(1, 'h'))
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
    lats = ds['lat'].values
    fill_north = False
    fill_south = False
    if lats[0] > lats[-1]:
        # Flip the latitude array
    #    ds['lat'] = lats[::-1]
        # Flip the data array
        ds = ds.sel(lat=slice(None, None, -1))
    # if -90. not in ds.lat.values:
    #     fill_south = True
    #     lat_values = np.insert(ds.lat.values, 0, -90)
    #     ds = ds.assign_coords(lat=lat_values)
    # if 90. not in ds.lat.values:
    #     fill_north = True
    #     lat_values = np.append(ds.lat.values, 90)
    #     ds = ds.assign_coords(lat=lat_values)

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
    ds['lev'] = levs
    if levs[0] < levs[-1]:
        # Flip the level array
    #    ds['lev'] = levs[::-1]
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

    ## Calculate ensemble mean
    if ens_mean:
        ds = ds.mean(dim="ens")

    ## Variables
    # rename variables
    rename_dict = {
        "10m_u_component_of_wind": "U10M",
        "10m_v_component_of_wind": "V10M",
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




    # define chunk size for each variable
    # mask variables based on elevation
    ## TODO : Need to check surface geopotential height from ERA5
    ## Get geopotential height 
    ds['H'] = ds['H']/MAPL_GRAV

    # topo
    topo = ds.PHIS.values/MAPL_GRAV
    height = ds.H.values
    mask = np.where(height > topo, 1, 0)
    for var in ds.data_vars:
        # add attributes
        ds[var].attrs = varMap[var]
        ds[var].attrs['_FillValue'] = FILL_VALUE
        ds[var].attrs['missing_value'] = FILL_VALUE
        ds[var].attrs['fmissing_value'] = FILL_VALUE
        # mask 3d variables
        if 'lev' in ds[var].dims:
            ds[var] = ds[var].where(mask == 1, FILL_VALUE)
    # chunk 
    #nlats = len(ds.lat)
    #nlons = len(ds.lon)
    #chunks_size = {"ens": 1, "time": 1, "lev": 1, "lat": nlats, "lon": nlons}
    #ds = ds.chunk(chunks_size)
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
    output_dir = Path(f"/discover/nobackup/projects/QEFM/data/rollout_outputs/{fmodel}/Y{yyyy}/M{mm}/D{dd}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{fmodel}-prediction-era5_date-{tstamp}_res-1.0_levels-13_ens-mean.nc"
    output_file = output_dir / fname
    ds.to_netcdf(output_file, encoding=encoding, engine="netcdf4")



    print("Finished \n", ds)
'''
