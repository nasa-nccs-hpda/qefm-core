print("start")
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import xarray as xr
import earthkit.data as ekd
import earthkit.regrid as ekr
from pathlib import Path
from anemoi.inference.runners.simple import SimpleRunner
#from anemoi.inference.outputs.printer import print_state
import argparse

import os
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

# argparse
parser = argparse.ArgumentParser(description="Run the AIFS model.")
parser.add_argument("--yyyy", "-y", type=str, default="2024", help="Year")
parser.add_argument("--mm", "-m", type=str, default="12", help="Month")
parser.add_argument("--dd", "-d", type=str, default="01", help="Day")

args = parser.parse_args()
YYYY = args.yyyy
MM = args.mm
DD = args.dd

# Set the default parameters
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_SFC_OUT = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]
SFC_LONG_NAME = ["10m_u_component_of_wind",
                 "10m_v_component_of_wind",
                 "2m_dewpoint_temperature",
                 "2m_temperature",
                 "mean_sea_level_pressure",
                 "skin_temperature",
                 "surface_pressure",
                 "total_column_water",
                 "land_sea_mask",
                 "geopotential_at_surface",
                 "slope_of_sub_gridscale_orography",
                 "standard_deviation_of_orography"]
PARAM_PL = ["z", "t", "u", "v", "w", "q"]
PL_LONG_NAME = ["geopotential",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
                "specific_humidity"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

DATE = datetime(int(YYYY), int(MM), int(DD), 0)
print("Initial date is", DATE)

def state_to_dataset(state):
    ## Convert the state to a dataset
    
    # Get the date from the state
    cdate = state["date"]
    # Datetime to string
    date_str = datetime.strftime(cdate, "%Y-%m-%dT%H")

    # Get the fields from the state
    fields = state.get("fields", {})
    #names = list(fields.keys())

    lats = np.linspace(90., -90., 721)
    lons = np.linspace(0., 359.75, 1440)
    # Convert the 2d state to a dataset
    ds_2d = xr.Dataset()
    for name in PARAM_SFC_OUT:
        # Create a DataArray for each parameter
        values = fields.get(name, None)
        if values is None:
            raise ValueError(f"Parameter {name} not found in the state.")
        else:
            values = interpolate(values, forward=False)
            da = xr.DataArray(values[None, :, :].astype(np.float32), 
                              dims=["time", "lat", "lon"], 
                              coords={"time": [cdate], "lat": lats.astype(np.float32), "lon": lons.astype(np.float32)},
                              name=name)
            # Add the DataArray to the dataset
            ds_2d[name] = da
    # Convert the 3d state to a dataset
    ds_3d = xr.Dataset()
    for name in PARAM_PL:
        vertical = []
        for l in LEVELS:
            name_lv = f"{name}_{l}"
            # Create a DataArray for each parameter

            values = fields.get(name_lv, None)
            if values is None:
                raise ValueError(f"Parameter {name_lv} not found in the state.")
            else:
                values = interpolate(values, forward=False)
                vertical.append(values)
        
        # Stack the vertical levels
        data = np.stack(vertical)
        da = xr.DataArray(data[None,:,:,:].astype(np.float32), 
                        dims=["time", "level", "lat", "lon"], 
                        coords={"time": [cdate], "level": np.array(LEVELS), "lat": lats.astype(np.float32), "lon": lons.astype(np.float32)},
                        name=name)
        # Add the DataArray to the dataset
        ds_3d[name] = da
    return xr.merge([ds_2d, ds_3d]), date_str

def interpolate(data, forward=True):
    if forward:
    # Interpolate the data from 0.25 to N320
        data = ekr.interpolate(data, {"grid": (0.25, 0.25)}, {"grid": "N320"})
    else:
        # Interpolate the data from N320 to 0.25
        data = ekr.interpolate(data, {"grid": "N320"}, {"grid": (0.25, 0.25)})
    return data

def get_nc_data(file, param, longname, levelist=[]):
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    ds = xr.open_dataset(file)
    for t in ds.time.values:
        for vs, vl in zip(param, longname):
            if vl not in ds.data_vars:
                raise ValueError(f"Variable {vl} not found in the dataset.")       
            var = ds[vl].sel(time=t)
            # Check if the variable is 3D or 2D
            if levelist:
                for lev in levelist:
                    f = var.sel(level=lev).squeeze().to_numpy()
                    values = interpolate(f)
                    # Add the values to the list
                    name = f"{vs}_{lev}"
                    fields[name].append(values)
            else:
                f = var.squeeze().to_numpy()
                values = interpolate(f)
                # Add the values to the list
                name = vs
                fields[name].append(values)
    for param, values in fields.items():
        fields[param] = np.stack(values)
    return fields


def get_open_data(param, levelist=[]):
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    for date in [DATE - timedelta(hours=6), DATE]:
        data = ekd.from_source("ecmwf-open-data", date=date, param=param, levelist=levelist)
        for f in data:
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721,1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25 to N320
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)
    for param, values in fields.items():
        fields[param] = np.stack(values)
    return fields

fields = {}
#file_path_os = "data.p"
input_root = Path("/discover/nobackup/projects/QEFM/data/FMAifs/nc_files")
file_path_os = input_root / f"aifs-dataset-source-era5_date-{YYYY}-{MM}-{DD}_res-0.25_levels-13_steps-0.nc"
#file_path_os = "/discover/nobackup/projects/QEFM/data/FMAifs/pkl_files/aifs-dataset-source-era5_date-2024-12-01_res-0.25_levels-13_steps-0.pkl"
#if os.path.exists(file_path_os):
#     print(f"File '{file_path_os}' exists.")
#     with open(file_path_os, 'rb') as fp:
#        fields = pickle.load(fp)
#     print("unpickled fields: \n", fields)
#else:
#     print(f"File '{file_path_os}' does not exist.")
#     fields.update(get_open_data(param=PARAM_SFC))
#     fields.update(get_open_data(param=PARAM_PL, levelist=LEVELS))
#     print("pickling ecmwf-open-data fields: \n", fields)
#     with open('data.p', 'wb') as fp:
#         pickle.dump(fields, fp, protocol=pickle.HIGHEST_PROTOCOL)
fields.update(get_nc_data(file_path_os, param=PARAM_SFC, longname=SFC_LONG_NAME))
fields.update(get_nc_data(file_path_os, param=PARAM_PL, longname=PL_LONG_NAME, levelist=LEVELS))
## out_path = "/discover/nobackup/projects/QEFM/data/FMAifs/pkl_files"
## date_str = datetime.datetime.strftime(DATE, "%Y-%m-%d")
## out_file_name = f"aifs-dataset-source-era5_date-{date_str}_res-0.25_levels-13_steps-0.pkl"
## out_file = os.path.join(out_path, out_file_name)
## with open(out_file, 'wb') as fp:
##    pickle.dump(fields, fp, protocol=pickle.HIGHEST_PROTOCOL)
## print("fields: \n", fields)

# for level in LEVELS:
#     gh = fields.pop(f"gh_{level}")
#     fields[f"z_{level}"] = gh * 9.80665

input_state = dict(date=DATE, fields=fields)

#checkpoint = {"huggingface":"ecmwf/aifs-single-0.2.1"}
checkpoint = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/aifs/aifs_single_v0.2.1.ckpt"
print(checkpoint)
runner = SimpleRunner(checkpoint, device="cuda")
#runner = SimpleRunner(checkpoint, device="cpu")

out_root = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/FMAifs/raw")
out_path = out_root / datetime.strftime(DATE, "%Y-%m-%d")
out_path.mkdir(parents=True, exist_ok=True)

for state in runner.run(input_state=input_state, lead_time=240):
    print("state at: \n", state.get("date"))
    #print_state(state)
    ds, str = state_to_dataset(state)
    out_file_name = f"prediction_date-{str}.nc"
    out_file = out_path / out_file_name
    ds.to_netcdf(out_file, mode="w", format="NETCDF4", engine="netcdf4")

