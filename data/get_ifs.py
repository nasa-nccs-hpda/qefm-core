from ecmwf.opendata import Client
from pathlib import Path
import numpy as np
import datetime
from collections import defaultdict

import earthkit.data as ekd
import earthkit.regrid as ekr


def get_open_data(param, cdate, levelist=[]):
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    for date in [cdate - datetime.timedelta(hours=6), cdate]:
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

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields

if "__name__" == "__main__":
    # Define the parameters and levels to retrieve
    # The parameters are defined in the IFS documentation
    PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
    PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
    LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    
    c = Client()
    DATE = c.latest()
    print("Latest date is", DATE)
    folder_name = DATE.strftime("%Y%m%dT%H")

    out_path = Path(f"/discover/nobackup/projects/QEFM/data/FMAifs/ifs_scda/{folder_name}")
    out_path.mkdir(parents=True, exist_ok=True)

    for step in range(8):
        date_step = DATE - datetime.timedelta(hours=step*6)
        file_name = out_path / f"IFS_{date_step.strftime('%Y%m%dT%H')}.npz"
        if file_name.exists():
            print(f"File {file_name} already exists, skipping")
            continue
        print(f"Processing file {file_name}")
        
        # Get the data for the current date and the previous date
        fields = {}
        fields.update(get_open_data(param=PARAM_SFC, cdate=DATE))
        fields.update(get_open_data(param=PARAM_PL, cdate=DATE, levelist=LEVELS))

        # Transform GH to Z safely
        for level in LEVELS:
            gh_key = f"gh_{level}"
            if gh_key in fields:
                gh = fields.pop(gh_key)
                fields[f"z_{level}"] = gh * 9.80665
            else:
                print(f"Warning: {gh_key} not found for {date_step}, skipping transformation.")

        # Save the data to a file
        np.savez(file_name, **fields)
 



