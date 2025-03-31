import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Convert GenCast output to CF-compliant NetCDF")
# parser.add_argument("input_dir", type=str, help="Path to GenCast output directory")
# parser.add_argument("fmodel", type=str, help="Model name")
parser.add_argument("--year", "-y", type=str, help="Year")
parser.add_argument("--month", "-m", type=str, help="Month")
parser.add_argument("--day", "-d", type=str, help="Day")
args = parser.parse_args()

input_dir = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/")
fmodel = "FMPrithvi-WxC"
yyyy = args.year
mm = args.month
dd = args.day
file_path = input_dir / fmodel / f"Y{yyyy}" / f"M{mm}" / f"D{dd}"

files = sorted(file_path.glob("pred_*.nc"))
#file = files[0]

MAPL_GRAV = 9.80665
FILL_VALUE = 1.e+15

for file in files:
    print("Processing file : ", file)
    ds = xr.open_dataset(file)
    print("At Open : \n", ds)
    ds.attrs = {}

    ## Expand dimensions for time
    #ds = ds.expand_dims(time=[1])

    ## Coordinates
    # Time
    base_name = file.name.split("_")
    date_str = base_name[2]
    hour_str = base_name[3][:2]
    dt = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{hour_str}:00:00", "ns")
    ds['time'] = dt

    long_name = "time"
    begin_date = f"{yyyy}{mm}{dd}"
    begin_time = 30000
    time_increment = 30000
    units = f"hours since {yyyy}-{mm}-{dd} 00:00:00"
    calendar = "proleptic_gregorian"

    # get time stamp for output file
    t = ds['time'].values
    tstamp = np.datetime_as_string(t, unit='h')
    print("Time stamp : ", tstamp)


    # change value of time
    ref_time = np.datetime64("2024-12-01T00:00:00", "ns")
    ds['time'] = (ds['time']-ref_time)/np.timedelta64(1, 'h')
    # add attributes
    ds.time.attrs = {
        "units" : units,
        "calendar" : calendar,
        "begin_date" : begin_date,
        "begin_time" : begin_time,
        "time_increment": time_increment
    }

    # Latitude
    lats = ds['lat'].values.astype(np.float32)
    fill_north = False
    if 90. not in ds.lat.values:
        fill_north = True
        lat_values = np.append(ds.lat.values, 90.0)
        #ds = ds.assign_coords(lat=lat_values)
    ds['lat'] = lats
    ds.lat.attrs = {
        "long_name" : "latitude",
        "units" : "degrees_north",
    }
    # Longitude
    lons = ds['lon'].values.astype(np.float32)
    ds['lon'] = lons
    ds.lon.attrs = {
        "long_name" : "longitude",
        "units" : "degrees_east",
    }

    # level
    #ds = ds.rename({'level': 'lev'})
    levs = ds['lev'].values.astype(np.float32)
    ds['lev'] = levs
    if levs[0] < levs[-1]:
        # Flip the data array
        ds = ds.sel(lev=slice(None, None, -1))
    ds.lev.attrs = {
        "long_name" : "model_level",
        "units" : "index",
    }
    print("After coord \n", ds)
    ## Variables
    # rename variables
    # rename_dict = {
    #     "u10": "U10",
    #     "v10": "V10",
    #     "t2m": "T2M",
    #     "z": "H",
    #     "msl": "SLP",
    #     "q": "QV",
    #     "t": "T",
    #     "u": "U",
    #     "v": "V",
    # }
    # ds = ds.rename(rename_dict)
    # print("After rename \n ", ds)



    # # map attributes
    # varMap = {
    #     "U10": {
    #         "long_name" : "10-meter_eastward_wind",
    #         "units" : "m s-1",
    #     },
    #     "V10": {
    #         "long_name" : "10-meter_northward_wind",
    #         "units" : "m s-1",
    #     },
    #     "T2M": {
    #         "long_name" : "2-meter_air_temperature",
    #         "units" : "K",
    #     },
    #     "H": {
    #         "long_name" : "height",
    #         "units" : "m",
    #     },
    #     "SLP": {
    #         "long_name" : "sea_level_pressure",
    #         "units" : "Pa",
    #     },
    #     "QV": {
    #         "long_name" : "specific_humidity",
    #         "units" : "kg kg-1",
    #     },
    #     "T": {
    #         "long_name" : "air_temperature",
    #         "units" : "K",
    #     },
    #     "U": {
    #         "long_name" : "eastward_wind",
    #         "units" : "m s-1",
    #     },
    #     "V": {
    #         "long_name" : "northward_wind",
    #         "units" : "m s-1",
    #     },
    #     "PHIS": {
    #         "long_name" : "surface_geopotential_height",
    #         "units" : "m+2 s-2",
    #     },
    # }

    # define chunk size for each variable
    # mask variables based on elevation
    ## TODO : Need to check surface geopotential height from ERA5
    # expand time dimension
    ds  = ds.expand_dims(dim={"time": 1})
    ones = ds.isel(lat=0).expand_dims(dim={"lat": 1})
    ones['lat'] = [np.float32(90.0)]
    ones = ones.map(lambda x: xr.full_like(x, FILL_VALUE))

    ds_new = xr.concat([ds, ones], dim="lat")
    # print(ds_new)
    # exit() 

    # # topo
    # topo = ds.PHIS.values/MAPL_GRAV
    # height = ds.H.values
    # mask = np.where(height > topo, 1, 0)
    # for var in ds.data_vars:
    #     arr = ds[var].values
    #     if len(arr.shape) == 3:
    #         fll = np.full((14, 1, 576), FILL_VALUE)
    #         data = np.concatenate((arr, fll), axis=1)
    #     elif len(arr.shape) == 2:
    #         fll = np.full((1, 576), FILL_VALUE)
    #         data = np.concatenate((arr, fll), axis=0)
    #     data_ex = data[None, ...]
    #     ds['var'] = xr.DataArray(data_ex, 
    #                              dims=('time', 'lev', 'lat', 'lon'),
    #                              coords={'time': ds.time, 'lev': ds.lev, 'lat': [lat_values], 'lon': ds.lon},)

    # #     # add attributes
    #     ds[var]= ds['var'].assign_coords({"time": ds.time})
    # #     # mask 3d variables
    #     if 'lev' in ds[var].dims:
    #         ds[var] = ds[var].where(mask == 1, FILL_VALUE)
    # # chunk 
    # # nlats = len(ds.lat)
    # # nlons = len(ds.lon)
    # # chunks_size = {"ens": 1, "time": 1, "lev": 1, "lat": nlats, "lon": nlons}
    # # ds = ds.chunk(chunks_size)
    print("After variable \n", ds_new)

    ## add global attributes
    ds_new.attrs = {
        "title" : f"{fmodel} forecast start at {yyyy}-{mm}-{dd}T00:00:00", 
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
    fname = f"{fmodel}-prediction-merra2_date-{tstamp}_res-0.5_levels-14.nc"
    output_file = output_dir / fname
    ds_new.to_netcdf(output_file, encoding=encoding, engine="netcdf4")



    print("Finished \n", ds_new)
