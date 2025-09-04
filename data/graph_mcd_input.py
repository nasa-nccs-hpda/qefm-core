## The script is used to generate the input data for GraphCast by merging the MCD data with the ERA5 data.
import os
import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe

def load_mcd_data(mcd_file):
    """Load MCD data from a NetCDF file."""
    return xr.open_mfdataset(mcd_file)

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
    orig_min = mcd_ds[var_name].min(dim=("lat", "lon"))
    orig_max = mcd_ds[var_name].max(dim=("lat", "lon"))
    target_min = era5_ds[var_name].min(dim=("lat", "lon"))
    target_max = era5_ds[var_name].max(dim=("lat", "lon"))
    scaled_data = (mcd_ds[var_name] - orig_min) / (orig_max - orig_min) * (target_max - target_min) + target_min
    return scaled_data


def main():
    mcd_root = "/discover/nobackup/projects/nccs_interns/mvu2/jli/data/revz/mcd_output_Ls285_hr00-rev-z.nc"
    era5_file = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/graphcast_dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
    output_file = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/source-era5-mcdv1_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

    hrs = ["00" , "06"]
    mcd_files = [mcd_root.replace("hr00", f"hr{hr}") for hr in hrs]
    mcd_ds = load_mcd_data(mcd_files)
    era5_ds = load_era5_data(era5_file)

    mcd_ds_preprocessed = preprocess_mcd_data(mcd_ds)
    print("MCD_processed")

    # Scale MCD variables to match ERA5 ranges
    swap_vars = ['toa_incident_solar_radiation']
    for var in swap_vars:
        if var in era5_ds.data_vars:
            mcd_ds_preprocessed[var] = scale_mcd_data(mcd_ds_preprocessed, era5_ds, var)
            # Replace ERA5 variable with MCD variable
            era5_ds[var][0,0:2,:,:] = mcd_ds_preprocessed[var][0:2,:,:]    

    # Save to NetCDF
    era5_ds.to_netcdf(output_file)
    print("After modification")
    print(era5_ds)

if __name__ == "__main__":
    main()

    