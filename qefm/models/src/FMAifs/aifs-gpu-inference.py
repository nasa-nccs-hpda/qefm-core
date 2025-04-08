print("start")
import datetime
from collections import defaultdict

import numpy as np
import xarray as xr
import earthkit.data as ekd
import earthkit.regrid as ekr

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

import os
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
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

DATE = datetime.datetime(2024, 12, 1, 0)
print("Initial date is", DATE)

def roll_and_interpolate(data):
    # Check if the data is in the expected shape
    if data.shape != (721, 1440):
        raise ValueError(f"Data shape is {data.shape}, expected (721, 1440)")
    # Shift the data from -180 to 180 to 0-360
    data = np.roll(data, -data.shape[1] // 2, axis=1)
    # Interpolate the data from 0.25 to N320
    data = ekr.interpolate(data, {"grid": (0.25, 0.25)}, {"grid": "N320"})
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
                    values = roll_and_interpolate(f)
                    # Add the values to the list
                    name = f"{vs}_{lev}"
                    fields[name].append(values)
            else:
                f = var.squeeze().to_numpy()
                values = roll_and_interpolate(f)
                # Add the values to the list
                name = vs
                fields[name].append(values)
    for param, values in fields.items():
        fields[param] = np.stack(values)
    return fields


def get_open_data(param, levelist=[]):
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    for date in [DATE - datetime.timedelta(hours=6), DATE]:
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
file_path_os = "/discover/nobackup/projects/QEFM/data/FMAifs/nc_files/aifs-dataset-source-era5_date-2024-12-01_res-0.25_levels-13_steps-0.nc"
# if os.path.exists(file_path_os):
#     print(f"File '{file_path_os}' exists.")
#     with open('data.p', 'rb') as fp:
#        fields = pickle.load(fp)
#     print("unpickled fields: \n", fields)
# else:
#     print(f"File '{file_path_os}' does not exist.")
#     fields.update(get_open_data(param=PARAM_SFC))
#     fields.update(get_open_data(param=PARAM_PL, levelist=LEVELS))
#     print("pickling ecmwf-open-data fields: \n", fields)
#     with open('data.p', 'wb') as fp:
#         pickle.dump(fields, fp, protocol=pickle.HIGHEST_PROTOCOL)
fields.update(get_nc_data(file_path_os, param=PARAM_SFC, longname=SFC_LONG_NAME))
fields.update(get_nc_data(file_path_os, param=PARAM_PL, longname=PL_LONG_NAME, levelist=LEVELS))
out_path = "/discover/nobackup/projects/QEFM/data/FMAifs/pkl_files"
out_file_name = f"aifs-dataset-source-era5_date-{datetime.strftime(DATE, "%Y-%m-%d")}_res-0.25_levels-13_steps-0.pkl"
out_file = os.path.join(out_path, out_file_name)
with open(out_file, 'wb') as fp:
    pickle.dump(fields, fp, protocol=pickle.HIGHEST_PROTOCOL)
print("fields: \n", fields)
exit()

# for level in LEVELS:
#     gh = fields.pop(f"gh_{level}")
#     fields[f"z_{level}"] = gh * 9.80665

input_state = dict(date=DATE, fields=fields)

checkpoint = {"huggingface":"ecmwf/aifs-single-0.2.1"}
print(checkpoint)
runner = SimpleRunner(checkpoint, device="cuda")
#runner = SimpleRunner(checkpoint, device="cpu")

for state in runner.run(input_state=input_state, lead_time=12):
    print_state(state)

