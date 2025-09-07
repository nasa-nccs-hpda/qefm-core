## The script is used to generate the input data for GraphCast by merging the MCD data with the ERA5 data.
import os
import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe
from datetime import datetime, timedelta

def load_mcd_data(mcd_file):
    """Load MCD data from a NetCDF file."""
    return xr.open_mfdataset(mcd_file, engine='netcdf4')

def load_era5_data(era5_file):
    """Load ERA5 data from a NetCDF file."""
    return xr.open_dataset(era5_file)

def regrid_mcd_data(mcd_ds, res=1.0):
    target_lat = np.arange(-90, 90 + res, res)
    target_lon = np.arange(0, 360, res)  # 0–360
    target_grid = xr.Dataset({'lat': (['lat'], target_lat),
                              'lon': (['lon'], target_lon)})
    
    regridder = xe.Regridder(mcd_ds, target_grid, 'bilinear',
                             periodic=True, ignore_degenerate=True)

    regridded_vars = {}
    for var in mcd_ds.data_vars:
        dims = mcd_ds[var].dims
        out_list = []
        for t in mcd_ds.time:
            sub = mcd_ds[var].sel(time=t)
            if dims == ("time", "lat", "lon"):
                regridded = regridder(sub)
            else:
                other_dims = [d for d in sub.dims if d not in ["lat", "lon"]]
                stacked = sub.stack(z=other_dims)
                regridded = regridder(stacked)
                regridded = regridded.unstack("z")
                regridded = regridded.transpose(*other_dims, "lat", "lon")
            out_list.append(regridded)
        regridded_vars[var] = xr.concat(out_list, dim="time")
        
    ds_out = xr.Dataset(regridded_vars, coords={"lat": target_lat, "lon": target_lon})
    return ds_out

def preprocess_mcd_data(mcd_ds):
    """Preprocess MCD data to match ERA5 resoltuion."""
    # Longitude adjustment -180~180 to 0~360
    if mcd_ds.lon.min() < 0:
        mcd_ds = mcd_ds.assign_coords(lon=(((mcd_ds.lon + 360) % 360)))
        mcd_ds = mcd_ds.sortby(mcd_ds.lon)

    # Regrid MCD data to 1-degree resolution
    mcd_ds_regridded = regrid_mcd_data(mcd_ds, res=1.0)
    return mcd_ds_regridded

def scale_mcd_data(mcd_ds, era5_ds, var_name):
    era5_sub = era5_ds[var_name].isel(time=slice(0, 2), batch=0)
    orig_min = mcd_ds[var_name].min(dim=("lat", "lon"))
    orig_max = mcd_ds[var_name].max(dim=("lat", "lon"))
    target_min = era5_sub.min(dim=("lat", "lon"))
    target_max = era5_sub.max(dim=("lat", "lon"))
    print(f"Scaling {var_name}: MCD min {orig_min}, max {orig_max}; ERA5 min {target_min}, max {target_max}")
    # Then index explicitly per timestep
    scaled_list = []
    for t in range(mcd_ds.sizes["time"]):
        scaled_list.append(
            (mcd_ds[var_name].isel(time=t) - orig_min[t]) /
            (orig_max[t] - orig_min[t]) *
            (target_max[t] - target_min[t]) +
            target_min[t]
        )
    scaled_data = xr.concat(scaled_list, dim="time")
    return scaled_data

def set_constants(era5_ds, var_name, value):
    if var_name in era5_ds.data_vars:
        era5_ds[var_name].values[:] = value
    return era5_ds

def constants_to_era5(era5_ds):

    era5_ds = set_constants(era5_ds, 'land_sea_mask', 1)
    era5_ds = set_constants(era5_ds, 'total_precipitation_6hr', 0.0)
    era5_ds = set_constants(era5_ds, 'geopotantial_at_surface', 25.0)
    return era5_ds

def main():
    graph_root = "/discover/nobackup/projects/QEFM/data/FMGenCast/6hr/samples"
    mcd_root = "/discover/nobackup/projects/nccs_interns/mvu2/jli/data/revz"
    #era5_file = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/graphcast_dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
    #output_root = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/source-era5-mcdv3_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
    output_root = "/discover/nobackup/projects/QEFM/data/FMGenCast/6hr/samples/mcd"

    n  = 3
    start_date = datetime(2022, 1, 1)
    dates = [(start_date + timedelta(days=5*i)).strftime('%Y-%m-%d') for i in range(n)]

    series = np.arange(285, 361, 5).tolist() + np.arange(0, 286, 5).tolist()
    for idx, Ls in enumerate(series[:n]):
        # Load MCD data
        mcd_scheme = os.path.join(mcd_root, f"mcd_output_Ls{Ls:02d}_hr00-rev-z.nc")

        hrs = ["00" , "06"]
        mcd_files = [mcd_scheme.replace("hr00", f"hr{hr}") for hr in hrs]
        mcd_ds = load_mcd_data(mcd_files)
        print(mcd_ds)

        # Load ERA5 data
        era5_file = os.path.join(graph_root, "graph", f"graphcast-dataset-source-era5_date-{dates[idx]}_res-1.0_levels-13_steps-4.nc")
        era5_ds = load_era5_data(era5_file)
        output_file = os.path.join(graph_root, "mcd", f"graphcast_dataset_source-era5-mcd_date-{dates[idx]}_res-1.0_levels-13_steps-4.nc")

        mcd_ds = mcd_ds.assign_coords(time=era5_ds.time.values[:2])
        mcd_ds_preprocessed = preprocess_mcd_data(mcd_ds)

        # Scale MCD variables to match ERA5 ranges
        swap_vars = ['2m_temperature', 'temperature']
        for var in swap_vars:
            if var in era5_ds.data_vars:
                mcd_ds_preprocessed[var] = scale_mcd_data(mcd_ds_preprocessed, era5_ds, var)
                print(mcd_ds_preprocessed[var].values)
                # Replace ERA5 variable with MCD variable
                era5_ds[var][0,0:2,:,:] = mcd_ds_preprocessed[var][0:2,:,:]    

        era5_ds = constants_to_era5(era5_ds)
        # Save to NetCDF
        era5_ds.to_netcdf(output_file)

if __name__ == "__main__":
    main()

    