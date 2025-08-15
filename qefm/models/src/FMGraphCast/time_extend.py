import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import datetime
from pathlib import Path
import argparse



print("Download and subset ERA5 input data:")
parser = argparse.ArgumentParser(description="Download and subset ERA5 input data:")
parser.add_argument("--outdir", "-o", default="/discover/nobackup/projects/QEFM/data/FMGraphCast/sim", type=str, help="Output directory")
parser.add_argument("--year", "-y", default="2024", type=str, help="Year of the data")
parser.add_argument("--month", "-m", default="12", type=str, help="Month of the data")
parser.add_argument("--day", "-d", default="01", type=str, help="Day of the data")
parser.add_argument("--freq", "-f", default="6", type=str, help="Frequency in hours")
parser.add_argument("--levs", "-l", default="37", type=str, help="Number of pressure levels")
parser.add_argument("--nsteps", "-n", default=42, type=str, help="Number of time steps")
parser.add_argument("--coarsen", "-c", default=False, type=bool, help="If True, coarsen the data to 1p0 degree")

args = parser.parse_args()
date_str = f"{args.year}-{args.month}-{args.day}"
nsteps = int(args.nsteps)
nlevs = int(args.levs)
start_time = f"{date_str}T00:00"
cfreq=f"{args.freq}"
output_dir=Path(f"{args.outdir}")
print("arguments:", args._get_kwargs)

def expand_dims(ds, steps):
    # Expand the time dimension of the dataset
    orig_time = ds.time.values
    extra_steps = steps-2

    # Assume a base time to convert timedelta to datetime
    #base_time = np.datetime64("2024-12-01")
    base_time = np.datetime64(date_str)

    print(base_time)
    abs_time = base_time + orig_time  # convert to datetime

    ds = ds.assign_coords(time=abs_time)  # update the dataset's time
    old_time = ds.time.values
    print(old_time)
    #################################################################
    ds_last = ds.isel(time=1)
    new_times = pd.date_range(start=old_time[-1], periods=extra_steps+1, freq="6h")[1:]

    repeated = ds_last.expand_dims(time=range(extra_steps)).copy(deep=True)
    repeated['time'] = new_times
    ds_extended = xr.concat([ds, repeated], dim='time')
    ds_extended['land_sea_mask'] = ds_extended['land_sea_mask'].isel(time=0)
    ds_extended['geopotential_at_surface'] = ds_extended['geopotential_at_surface'].isel(time=0)
    print(ds_extended['land_sea_mask'], ds_extended['geopotential_at_surface'])
    return ds_extended

if __name__ == "__main__":
   #file = 'graphcast-dataset-source-era5_date-_date_str res-0.25_levels-37_freq-6h_steps-3.nc'
   file = f"/discover/nobackup/projects/QEFM/data/FMGraphCast/6h/Y2024/graphcast-dataset-source-era5_date-{date_str}_res-0.25_levels-37_freq-6h_steps-3.nc"
   print("input: ", file)
   out_file=f"{args.outdir}/graphcast-dataset-source-era5_date-{str(date_str)}_res-0.25_levels-37_freq-6h_steps-{str(nsteps)}.nc"
   print("output: ", out_file)
   ds_short = xr.open_dataset(file)

   ds_long = expand_dims(ds_short, nsteps)
   print(f"Size of time dimension : {ds_long.sizes['time']}")
   ds_long.to_netcdf(out_file)
   print("Wa la...:", out_file)
