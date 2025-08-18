import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

print("Convert GraphCast output to CF-compliant NetCDF")
parser = argparse.ArgumentParser(description="Convert GraphCast output to CF-compliant NetCDF")
parser.add_argument("--indir", default="/discover/nobackup/projects/QEFM/data/FMGraphCast/rollout_outputs", type=str, help="Path to GraphCast rollout directory")
parser.add_argument("--outdir", default="/discover/nobackup/projects/QEFM/data/rollout_outputs/FMGraphCast/v20250815", type=str, help="Path to GraphCast CF output directory")
parser.add_argument("--fmodel", default="FMGraphCast", type=str, help="Model name")
parser.add_argument("--year", "-y", default="2024", type=str, help="Year of the data")
parser.add_argument("--month", "-m", default="12", type=str, help="Month of the data")
parser.add_argument("--day", "-d", default="01", type=str, help="Day of the data")
parser.add_argument("--tsteps", "-t", default="41", type=str, help="Number of time steps")

args = parser.parse_args()
print("arguments:", args._get_kwargs)

input_dir = Path(args.indir)
out_dir = Path(args.outdir)
fmodel = args.fmodel
file_path = input_dir
yyyy = args.year
mm = args.month
dd = args.day
tsteps = args.tsteps

files = sorted(file_path.glob(f"*{yyyy}-{mm}-{dd}_*{tsteps}.nc"))
file = files[0]
print(file_path,file)

MAPL_GRAV = 9.80665
FILL_VALUE = np.float32(1.e+15)
ds_org = xr.open_dataset(file)
ref_date = np.datetime64(f"{yyyy}-{mm}-{dd}T06:00:00")

## Coordinates
# For GenCast Only, remove "batch"
ds_org = ds_org.squeeze(dim="batch")

# add variable geopotential at surface
tmp_file=f"/discover/nobackup/projects/QEFM/data/FMGraphCast/sim/graphcast-dataset-source-era5_date-2024-12-{dd}_res-0.25_levels-37_freq-6h_steps-{str(int(tsteps)+1)}.nc"
#source = Path("/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/6h/_Y2024")
#tmp_file = list(source.glob(f"*{yyyy}-{mm}-{dd}_*.nc"))[0]
print("tmp_file: ", tmp_file)
ds_temp = xr.open_dataset(tmp_file)
ds_org['PHIS'] = ds_temp['geopotential_at_surface']
#ds_org['PHIS'] = ds_temp['geopotential']
ens_mean = False

times = ds_org.time.values
for ctime in times:
    ds = ds_org.sel(time=ctime).expand_dims("time")
    #print(ds.variables['time'])
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
    time_increment = np.int32(60000)
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

    # # ensemble
    # ds = ds.rename({'sample': 'ens'})
    # ds.ens.attrs = {
    #     "long_name" : "ensemble_member",
    #     "units" : " ",
    # }
    # print("After coord \n", ds['time'])

    # ## Calculate ensemble mean
    # if ens_mean:
    #     ds = ds.mean(dim="ens")

    ## Variables
    # rename variables
    rename_dict = {
        "10m_u_component_of_wind": "U10M",
        "10m_v_component_of_wind": "V10M",
        "2m_temperature": "T2M",
        "geopotential": "H",
        "mean_sea_level_pressure": "SLP",
#        "sea_surface_temperature": "SST",
        "specific_humidity": "QV",
        "temperature": "T",
        "total_precipitation_6hr": "PRECTOT",
        "u_component_of_wind": "U",
        "v_component_of_wind": "V",
        "vertical_velocity": "OMEGA",
#        "geopotential_at_surface": "PHIS",
    }
    ds = ds.rename(rename_dict)
    #print("After rename \n ", ds)


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

    # define chunk size for each variable
    # mask variables based on elevation
    ## TODO : Need to check surface geopotential height from ERA5
    ## Get geopotential height 
    ds['H'] = ds['H']/MAPL_GRAV

    # topo
    topo = ds_org['PHIS'].values/MAPL_GRAV
    # topo = ds.PHIS.values/MAPL_GRAV
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
    #print("After variable \n", ds)

    ## add global attributes
    ds.attrs = {
        "title" : f"{fmodel} forecast start at {yyyy}-{mm}-{dd}T06:00:00", 
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
    output_dir = Path(f"{out_dir}/{fmodel}/Y{yyyy}/M{mm}/D{dd}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{fmodel}-prediction-era5_date-{tstamp}_res-0.25_levels-37_cf.nc"
    output_file = output_dir / fname
    ds.to_netcdf(output_file, encoding=encoding, engine="netcdf4")



    print("Finished \n", output_file)
