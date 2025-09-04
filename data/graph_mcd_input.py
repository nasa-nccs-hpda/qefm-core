## The script is used to generate the input data for GraphCast by merging the MCD data with the ERA5 data.
import os
import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe

def load_mcd_data(mcd_file):
    """Load MCD data from a NetCDF file."""
    return xr.open_dataset(mcd_file)

def load_era5_data(era5_file):
    """Load ERA5 data from a NetCDF file."""
    return xr.open_dataset(era5_file)

def regrid_mcd_data(mcd_ds, res=1.0):
    target_lat = np.arange(-90, 90 + res, res)
    target_lon = np.arange(-180, 180 + res, res)
    target_grid = xr.Dataset({'lat': (['lat'], target_lat),
                              'lon': (['lon'], target_lon)})
    
    regridder = xe.Regridder(mcd_ds, target_grid, 'bilinear', periodic=True, reuse_weights=True, ignore_degenerate=True)

    regridded_vars = {}

    for var in mcd_ds.data_vars:
        dims = mcd_ds[var].dims
        if "lat" in dims and "lon" in dims:
            # If 2D
            if dims == ("lat", "lon"):
                regridded_vars[var] = regridder(mcd_ds[var])
            # If 3D+ (e.g., level, lat, lon)
            else:
                # Stack non-lat/lon dims
                other_dims = [d for d in dims if d not in ["lat", "lon"]]
                stacked = ds[var].stack(z=other_dims)  # shape: (z, lat, lon)
                regridded = regridder(stacked)  # shape: (z, new_lat, new_lon)
                unstacked = regridded.unstack("z")
                regridded_vars[var] = unstacked.transpose(*other_dims, "lat", "lon")


    ds_out = xr.Dataset(regridded_vars, coords={"lat": target_lat, "lon": target_lon})
    return ds_out

def preprocess_mcd_data(mcd_ds):
    """Preprocess MCD data to match ERA5 resoltuion."""
    # Example: Regrid MCD data to 1-degree resolution
    mcd_ds_regridded = regrid_mcd_data(mcd_ds, res=1.0)
    return mcd_ds_regridded

def scale_mcd_data(mcd_ds, era5_ds, var_name):
    orig_min = mcd_ds[var_name].min().item()
    orig_max = mcd_ds[var_name].max().item()
    target_min = era5_ds[var_name].min().item()
    target_max = era5_ds[var_name].max().item()
    scaled_data = (mcd_ds[var_name] - orig_min) / (orig_max - orig_min) * (target_max - target_min) + target_min
    return scaled_data


def main():
    mcd_file = "/discover/nobackup/projects/nccs_interns/mvu2/jli/data/revz/mcd_output_Ls285_hr12-rev-z.nc"
    era5_file = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/graphcast_dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
    output_file = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/source-era5-mcdv1_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

    mcd_ds = load_mcd_data(mcd_file)
    print("MCD")
    print(mcd_ds)
    era5_ds = load_era5_data(era5_file)
    print("ERA5")
    print(era5_ds)

    mcd_ds_preprocessed = preprocess_mcd_data(mcd_ds)

    # Scale MCD variables to match ERA5 ranges
    swap_vars = ['2m_temperature', 'toa_solar_radiation']
    for var in swap_vars:
        if var in era5_ds.data_vars:
            mcd_ds_preprocessed[var] = scale_mcd_data(mcd_ds_preprocessed, era5_ds, var)
            # Replace ERA5 variable with MCD variable
            era5_ds[var][0,:,:,:] = mcd_ds_preprocessed[var][0,:,:]    

    # Save to NetCDF
    #combined_ds.to_netcdf(output_file)

if __name__ == "__main__":
    main()

    