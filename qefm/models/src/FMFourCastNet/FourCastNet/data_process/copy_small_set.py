import h5py
from netCDF4 import Dataset as DS
import numpy as np
import time
import os
import os
import time
import h5py
from netCDF4 import Dataset as DS

def writetofile(src, dest, channel_idx, varslist, src_idx=0, frmt='nc'):
    """
    Reads a variable from a source NetCDF or HDF5 file and writes it to a destination HDF5 file.
    Runs in serial (without MPI) and processes all data at once (no batching).
    """
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Source file not found: {src}")

    for variable_name in varslist:
        print(f"Processing variable: {variable_name}")

        # Open source file
        if frmt == 'nc':
            with DS(src, 'r') as fsrc:
                var_data = fsrc.variables[variable_name][:]
                vshape = var_data.shape
        elif frmt == 'h5':
            with h5py.File(src, 'r') as fsrc:
                var_data = fsrc[variable_name][:]
                vshape = var_data.shape

        # If 4D, select the specified src_idx
        if len(vshape) == 4:
            var_data = var_data[:, src_idx]  # Extract data from src_idx

        # Open destination file and write data
        with h5py.File(dest, 'a') as fdest:
            start = time.time()
        # Ensure 'fields' dataset exists
            if "fields" not in fdest:
                print("Creating 'fields' dataset in destination file...")
                shape = (var_data.shape[0], 21, var_data.shape[1], var_data.shape[2])  # Adjust as needed
                maxshape = (None, 21, var_data.shape[1], var_data.shape[2])  # Allow unlimited growth along axis 0
                fdest.create_dataset("fields", shape=shape, maxshape=maxshape, dtype=var_data.dtype)
            
            fdest['fields'][:, channel_idx, :, :] = var_data
            print(f"Finished {variable_name} in {time.time() - start:.2f} sec")


# Paths
dest = '/discover/nobackup/jli30/data/FNO/jan_2025_01_05.h5'

# Process different variables

src = '/discover/nobackup/jli30/data/FNO/ERA5/jan_2025_01_05_sfc.nc'
# u10 v10 t2m sp mslp
writetofile(src, dest, 0, ['u10'])
writetofile(src, dest, 1, ['v10'])
writetofile(src, dest, 2, ['t2m'])
writetofile(src, dest, 3, ['sp'])
writetofile(src, dest, 4, ['msl'])

src = '/discover/nobackup/jli30/data/FNO/ERA5/jan_2025_01_05_pl.nc'
# t850
writetofile(src, dest, 5, ['t'], 2)
# uvz1000
writetofile(src, dest, 6, ['u'], 3)
writetofile(src, dest, 7, ['v'], 3)
writetofile(src, dest, 8, ['z'], 3)
# uvz850
writetofile(src, dest, 9, ['u'], 2)
writetofile(src, dest, 10, ['v'], 2)
writetofile(src, dest, 11, ['z'], 2)
# uvz500
writetofile(src, dest, 12, ['u'], 1)
writetofile(src, dest, 13, ['v'], 1)
writetofile(src, dest, 14, ['z'], 1)
# t500
writetofile(src, dest, 15, ['t'], 1)
#z50
writetofile(src, dest, 16, ['z'], 0)
# r500
writetofile(src, dest, 17, ['r'], 1)
# r850
writetofile(src, dest, 18, ['r'], 2)

# tcwv
src = '/discover/nobackup/jli30/data/FNO/ERA5/jan_2025_01_05_sfc.nc'
writetofile(src, dest, 19, ['tcwv'])
