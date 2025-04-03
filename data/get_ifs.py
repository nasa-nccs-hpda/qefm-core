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

    fields = {}
    c = Client()
    DATE = c.latest()
    print("Latest date is", DATE)

    # Get the data for the current date and the previous date
    fields.update(get_open_data(param=PARAM_SFC, cdate=DATE))
    fields.update(get_open_data(param=PARAM_PL, cdate=DATE, levelist=LEVELS))

    # Transform GH to Z
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    # Save the data to a file
    path = Path("/discover/nobackup/projects/QEFM/data/FMAifs/ifs_scda")
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / f"IFS_{DATE.strftime('%Y%m%d')}.npz", **fields)


