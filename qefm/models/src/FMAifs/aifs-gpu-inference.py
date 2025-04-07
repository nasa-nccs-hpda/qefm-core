print("start")
import datetime
from collections import defaultdict

import numpy as np
import earthkit.regrid as ekr

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

import os
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

DATE = datetime.datetime(2025, 3, 20, 6)
print("Initial date is", DATE)

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
file_path_os = "data.p"
if os.path.exists(file_path_os):
    print(f"File '{file_path_os}' exists.")
    with open('data.p', 'rb') as fp:
       fields = pickle.load(fp)
    print("unpickled fields: \n", fields)
else:
    print(f"File '{file_path_os}' does not exist.")
    fields.update(get_open_data(param=PARAM_SFC))
    fields.update(get_open_data(param=PARAM_PL, levelist=LEVELS))
    print("pickling ecmwf-open-data fields: \n", fields)
    with open('data.p', 'wb') as fp:
        pickle.dump(fields, fp, protocol=pickle.HIGHEST_PROTOCOL)

for level in LEVELS:
    gh = fields.pop(f"gh_{level}")
    fields[f"z_{level}"] = gh * 9.80665

input_state = dict(date=DATE, fields=fields)

checkpoint = {"huggingface":"ecmwf/aifs-single-0.2.1"}
print(checkpoint)
runner = SimpleRunner(checkpoint, device="cuda")
#runner = SimpleRunner(checkpoint, device="cpu")

for state in runner.run(input_state=input_state, lead_time=12):
    print_state(state)

