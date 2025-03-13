# Description: This script is for Prithvi rollout prediciton only
# To converts the numpy arrays to netcdf files which share the same attributes of MERRA2.

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

surface_vars = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M",
]

vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]

levels = [
    34.0,
    39.0,
    41.0,
    43.0,
    44.0,
    45.0,
    48.0,
    51.0,
    53.0,
    56.0,
    63.0,
    68.0,
    71.0,
    72.0,
]

def get_surf_template(m2_path: str, init_time: str):
    surf_file = Path(m2_path) / f"MERRA2_sfc_{init_time}.nc"
    if not surf_file.exists():
        raise FileNotFoundError(f"The surface file {surf_file} does not exist.")
    #ds = xr.open_dataset(surf_file, engine="netcdf4")
    ds = xr.open_dataset("/discover/nobackup/projects/QEFM/data/FMPrithvi-WxC/merra-2/MERRA2_sfc_20241201.nc")
    return ds.isel(time=0, lat=slice(None, -1))

def get_pres_template(m2_path: str, init_time: str):
    pres_file = Path(m2_path) / f"MERRA2_pres_{init_time}.nc"
    if not pres_file.exists():
        raise FileNotFoundError(f"The pressure file {pres_file} does not exist.")
    ds = xr.open_dataset(pres_file, engine="netcdf4") 
    return ds.isel(time=0, lat=slice(None, -1), lev=slice(None, None, -1))

def arr_to_ds(data, template, time_value, var_names, surf=True):
    template = template.assign_coords(time=time_value)
    for i, var in enumerate(var_names):
        if surf:
            template[var].data = data[i].squeeze()
        else:
            idx0=len(surface_vars)+i*len(levels)
            idx1=len(surface_vars)+(i+1)*len(levels)        
            template[var].data = data[idx0:idx1].squeeze()
    return template


def write_to_netcdf(outputs: list[np.ndarray], m2_path: str, out_root: str, init_time: str):
    # Create the output directory
    out_dir = Path(out_root) / f"Y{init_time[:4]}/M{init_time[4:6]}/D{init_time[6:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create the time array
    time_stamps = pd.date_range(f"{init_time}T06", periods=len(outputs), freq='3h')

    # Get the templates
    surf_template = get_surf_template(m2_path, init_time)
    pres_template = get_pres_template(m2_path, init_time)

    # Write the output at each time step
    for i, output in enumerate(outputs):
        time = time_stamps[i]
        odata = output.detach().cpu().numpy() if hasattr(output, "detach") else np.array(output)
        odata = odata.squeeze()
        print(odata.shape)
        surf_ds = arr_to_ds(odata, surf_template, time, surface_vars)
        pres_ds = arr_to_ds(odata, pres_template, time, vertical_vars, surf=False)
        ds = xr.merge([surf_ds, pres_ds])
        
        fn = f"pred_prithvi_{time.strftime('%Y%m%d')}_{time.strftime('%H')}z.nc"
        out_path = out_dir / fn
        ds.to_netcdf(out_path)
        print(f"Saved {out_path}")
