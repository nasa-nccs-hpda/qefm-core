# this script has been modified to read era-5 data from /css/era5/pressure_level_hourly instead of downloading the data locally
# as of 2/12/2025, only the atmospheric variables are downloaded to /css/era5. There are directories for static and surface data, but they are empty. For now, download these locally


#https://microsoft.github.io/aurora/example_era5.html#loading-and-running-the-model



#import cdsapi
from pathlib import Path
import glob

data_path = Path("/css/era5")
#data_path = Path("/discover/nobackup/jli30/data/Aurora")
output_path = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/FMAurora")
#output_path = Path("/discover/nobackup/khbreen/qefm_local/rollout_outputs/FMAurora")
#data_path = Path("/discover/nobackup/khbreen/qefm_local/tmp")

import xarray as xr
from datetime import datetime, timedelta
import numpy as np
import time


import torch
from aurora import Batch, Metadata
from aurora import Aurora, rollout

model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
model.load_checkpoint_local("/discover/nobackup/khbreen/qefm_local/qefm-core/qefm/models/src/FMAurora/aurora-0.25-pretrained.ckpt")
model.eval()
model = model.to("cuda")

'''
# these are the "standard names" for the variables we want. stnard_name is an attribute in the *.nc files
static_vars = ["geopotential", "land_sea_mask", "soil_type"]
surf_vars = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]
atmos_vars = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
'''

def download_data(root_dir, start_date, end_date, stat_lst, surf_lst, atmos_lst):
    c = cdsapi.Client()
    current_date = start_date
    year = f"Y{current_date.year}"
    month = f"M{current_date.month:02d}"

    # get surface data for each time step
    # instantaneous surface data (as opposed to mean/max/min, column integrated, etc) 
    while current_date <= end_date:
          
        surf_dir_path = Path(root_dir) / "surface_hourly" / "inst" / year / month
        surf_dir_path.mkdir(parents=True, exist_ok=True)
        
        surf_file_name = f"era5_surface-inst_allvar_{current_date.strftime('%Y%m%d_%H')}z.nc"
        surf_file_path = surf_dir_path / surf_file_name  
        print('[]---------',surf_file_path)
        
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': surf_lst,
                    'year': str(current_date.year),
                    'month': f"{current_date.month:02d}",
                    'day': f"{current_date.day:02d}",
                    'time': f"{current_date.hour:02d}:00",
                    'format': 'netcdf'
                },
                str(surf_file_path)
            )
            print(f"Saved: {surf_file_path}")
        except Exception as e:
            print(f"Failed to download SURFACE data for {current_date}: {e}")
    
        
        # get atmospheric data for each time step
        # instantaneous surface data (as opposed to mean/max/min, column integrated, etc) 
        
        atmos_dir_path = Path(root_dir) / "pressure_hourly" / "inst" / year / month
        atmos_dir_path.mkdir(parents=True, exist_ok=True)
        
        atmos_file_name = f"era5_atmos-inst_allvar_{current_date.strftime('%Y%m%d_%H')}z.nc"
        atmos_file_path = atmos_dir_path / atmos_file_name 
        print('[]--------------',atmos_file_path) 
        
        try:
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': atmos_lst,
                    'year': str(current_date.year),
                    'month': f"{current_date.month:02d}",
                    'day': f"{current_date.day:02d}",
                    'time': f"{current_date.hour:02d}:00",
                    'pressure_level': [
                                      "1", "2", "3",
                                      "5", "7", "10",
                                      "20", "30", "50",
                                      "70", "100", "125",
                                      "150", "175", "200",
                                      "225", "250", "300",
                                      "350", "400", "450",
                                      "500", "550", "600",
                                      "650", "700", "750",
                                      "775", "800", "825",
                                      "850", "875", "900",
                                      "925", "950", "975",
                                      "1000"
                                      ],
                    'format': 'netcdf'
                },
                str(atmos_file_path)
            )
            print(f"Saved: {atmos_file_path}")
        except Exception as e:
            print(f"Failed to download ATMOS data for {current_date}: {e}")
        
        current_date += timedelta(hours=6)

fill_value = 1.e+15
surf_var_dict = {
        "2t": {
            "var_name": "T2M",
            "units": "K",
            "_FillValue": fill_value,
            "long_name": "2 metre temperature"
            
            },
        "10u": {
            "var_name": "U10M",
            "units": "m s**-1",
            "_FillValue": fill_value,
            "long_name": "10 metre U wind component"
            },
        "10v": {
            "var_name": "V10M",
            "units": "m s**-1",
            "_FillValue": fill_value,
            "long_name": "10 metre V wind component"
            },
        "msl": {
            "var_name": "SLP",
            "units": "Pa",
            "_FillValue": fill_value,
            "long_name": "Mean sea level pressure"
            }
        }
    
static_var_dict = {
        "z": {
            "var_name": "PHIS",
            "units": "m**2 s**-2",
            "_FillValue": fill_value,
            "long_name": "geopotential"
            },
        "slt": {
            "var_name": "SLT",
            "units": "(Code table 4.213)",
            "_FillValue": fill_value,
            "long_name": "Soil type"
            },
        "lsm": {
            "var_name": "LSM",
            "units": "(0 - 1)",
            "_FillValue": fill_value,
            "long_name": "Land-sea mask"
            }
        }
    
atmos_var_dict = {
        "t": {
            "var_name": "T",
            "units": "K",
            "_FillValue": fill_value,
            "long_name": "air_temperature"
            },
        "u": {
            "var_name": "U",
            "units": "m s**-1",
            "_FillValue": fill_value,
            "long_name": "U component of wind"
            },
        "v": {
            "var_name": "V",
            "units": "m s**-1",
            "_FillValue": fill_value,
            "long_name": "V component of wind"
            },
        "q": {
            "var_name": "QV",
            "units": "kg kg**-1",
            "_FillValue": fill_value,
            "long_name": "Specific humidity"
            },
        "z": {
            "var_name": "Z",
            "units": "m**2 s**-2",
            "_FillValue": fill_value,
            "long_name": "geopotential"
            }
        }
    
    
def _np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()    
    
def to_fluid(predvar, filepath):
        """Write the batch to a file.

        This requires `xarray` and `netcdf4` to be installed.
        """
        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("`xarray` must be installed.") from e
        
        #print('==== predvar',predvar.metadata.time,type(predvar.metadata.time))
        #print(np.array(list(predvar.metadata.time)), type(np.array(list(predvar.metadata.time))))
        
        #for k, v in predvar.surf_vars.items():
        #    print("[]----",k)
        #    print("    **    ",predvar.surf_vars[k].shape)
        #    print("        ****    ",predvar.surf_vars[k].squeeze(dim=0).shape,'\n\n')
        
        ds = xr.Dataset(
            {
                **{
                    surf_var_dict[k]["var_name"]: (("time", "lat", "lon"), _np(v.squeeze(dim=0)), {"units": surf_var_dict[k]["units"], "long_name": surf_var_dict[k]["long_name"]})
                    for k, v in predvar.surf_vars.items()
                },
                **{
                    static_var_dict[k]["var_name"]: (("lat", "lon"), _np(v), {"units": static_var_dict[k]["units"], "long_name": static_var_dict[k]["long_name"]})
                    for k, v in predvar.static_vars.items()
                },
                **{
                    atmos_var_dict[k]["var_name"]: (("time", "lev", "lat", "lon"), _np(v.squeeze(dim=0)), {"units": atmos_var_dict[k]["units"], "long_name": atmos_var_dict[k]["long_name"]})
                    for k, v in predvar.atmos_vars.items()
                },
            },
            coords={
                "time": ("time", list(predvar.metadata.time)),# {"units": "seconds since 1970-01-01", "calendar": "proleptic_gregorian", "standard_name": "time", "long_name": "time"}),
                "lev": ("lev", list(predvar.metadata.atmos_levels), {"units": "hPa", "standard_name": "air_pressure", "long_name": "pressure", "stored_direction": "decreasing", "positive": "down"}),
                "lat": ("lat", _np(predvar.metadata.lat), {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude", "stored_direction": "decreasing"}),
                "lon": ("lon", _np(predvar.metadata.lon), {"units": "degrees_east", "standard_name": "longitude", "long_name": "longitude"})
            },
        )
        #print("====",ds)
        encx = dict(dtype= 'float32', _FillValue= fill_value)
        enc = {var: encx for var in ds.data_vars}
        ds.to_netcdf(filepath, mode="w", encoding=enc, format="NETCDF4") 
        

# list variable names to be read in 
static_vars = ["z", "lsm", "slt"]
surf_vars = ["t2m", "u10", "v10", "msl"]
atmos_vars = ["t", "u", "v", "q", "z"]
plevs = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# path to read in - this will point to a single *.nc file for a given YYYYMMDD_HHz
# as of 2/12/25, there are data from 11/2018 - 12/2024 on Discover

start_date = datetime(2024, 5, 7, 0, 0)
end_date = datetime(2024, 5, 31, 0, 0)
current_date = start_date
t0 = time.time()
while current_date <= end_date:

    print("[]---- processing ", current_date)
    t1 = time.time()

    yyyy = str(current_date.year)
    mm = f"{current_date.month:02d}"
    dd = f"{current_date.day:02d}"
    HH = f"{current_date.hour:02d}:00"  # can be 00, 06, 12, or 18
    year = f"Y{current_date.year}"
    month = f"M{current_date.month:02d}"
    day = f"D{current_date.day:02d}"

    static_filename = f"era5_static-allvar.nc"
    stat_file_path = data_path / "static" / static_filename
    static_vars_ds = xr.open_dataset(stat_file_path, engine="netcdf4")[static_vars]#.squeeze()
    #stat_file_path = data_path / "static.nc"
    #static_vars_ds = xr.open_dataset(stat_file_path, engine="netcdf4")
    #print("==== STATIC ERA5 ",static_vars_ds)
    
    surf_filename = f"era5_surface-inst_allvar_{yyyy}{mm}{dd}_*.nc"
    f"era5_surface-inst_allvar_{current_date.strftime('%Y%m%d_%H')}z.nc"
    surf_file_path = data_path / "surface_hourly" / "inst" / year / month / surf_filename
    if not surf_file_path.exists():
        download_data("./", current_date, start_date+timedelta(days=1), static_lst, surface_lst, atmospheric_lst)
        surf_file_path = Path(".") / surf_filename 
    surf_file_list = glob.glob(str(surf_file_path))
    #print('[]----surf_file_list',surf_file_list)
    surf_vars_ds = xr.open_mfdataset(surf_file_list, engine="netcdf4")[surf_vars]#.squeeze()
    #surf_file_path = data_path / f"{yyyy}-{mm}-{dd}-surface-level.nc"
    #surf_vars_ds = xr.open_dataset(surf_file_path, engine="netcdf4")
    #print("==== SURFACE ERA5 ", surf_vars_ds)
    
    atmos_filename = f"era5_atmos-inst_allvar_{yyyy}{mm}{dd}_*.nc"
    atmos_file_path = data_path / "pressure_hourly" / "inst" / year / month / atmos_filename
    atmos_file_list = glob.glob(str(atmos_file_path))
    atmos_vars_ds = xr.open_mfdataset(atmos_file_list, engine="netcdf4")[atmos_vars].sel(pressure_level=plevs)#.squeeze()
    #atmos_file_path = data_path / f"{yyyy}-{mm}-{dd}-atmospheric.nc"
    #atmos_vars_ds = xr.open_dataset(atmos_file_path, engine="netcdf4")
    #print("==== ATMOS ERA5 ", atmos_vars_ds)
    
    #print("Preparing a Batch")

    # this structure is written to read inputs with the shape (time, lat, lon) or (time, lev, lat, lon) where time > 1. the variable idx is the index of the last time step that you want to subset for prediction.
    i = 0  # Select this time index in the downloaded data.

    batch = Batch(
        surf_vars={
            # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
            # batch dimension of size one.
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i - 1, i]][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i - 1, i]][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i - 1, i]][None]),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[i - 1, i]][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[i - 1, i]][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[i - 1, i]][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[i - 1, i]][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[i - 1, i]][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
        )

    # write 00z initial state to file
    #print('[]----- batch')#,initial_state)
    #for k,v in batch.surf_vars.items():
    #  print(k)
    #  print(v.shape)
      
    initial_state = Batch(
                        surf_vars={k: v[:, 0:1, ...] for k, v in batch.surf_vars.items()},  # First batch of surface variables
                        static_vars=batch.static_vars,  # Static vars don't have a batch dimension
                        atmos_vars={k: v[:, 0:1, ...] for k, v in batch.atmos_vars.items()},  # First batch of atmospheric variables
                        metadata=batch.metadata  # Metadata is the same for all batches
    )
    #print('[]----- initial state')#,initial_state)
    #for k,v in initial_state.surf_vars.items():
    #  print(k)
    #  print(v.shape)
    write_folder = output_path / year / month / day 
    # Create the folder if it doesn't exist
    write_folder.mkdir(parents=True, exist_ok=True)
    write_path = write_folder / f"init_aurora_{current_date.strftime('%Y%m%d_%H')}z.nc"
    to_fluid(initial_state, write_path)   

    #print("Loading and Running the Model")

    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=40-1)] 
    
    
    # write to netcdf   
           
    #Write prediction to NetCDF file
    # uses the .to_netcdf() method for the Batch class defined in /aurora/batch.py
    pred_date = current_date
    for i in range(len(preds)):
        pred = preds[i]
        pred_date += timedelta(hours=6)
        write_folder = output_path / year / month / day 
        # Create the folder if it doesn't exist
        write_folder.mkdir(parents=True, exist_ok=True) 
        write_path = write_folder / f"pred_aurora_{pred_date.strftime('%Y%m%d_%H')}z.nc"
        to_fluid(pred, write_path)   

    t2 = time.time()
    t_time = (t2-t1)/60
    total_time = (t2-t0)/60
    print(f"    **** time to process {current_date}: {t_time} MINUTES")
    print(f"--------**** TOTAL TIME: {total_time} MINUTES--------\n\n")
    
    current_date += timedelta(days=1)

