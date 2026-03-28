from datetime import datetime, timedelta
from merra21c import MERRA21cObsConv, MERRA21cObsSat

# Conventional observations - use MERRA21cObsConv
conv_ds = MERRA21cObsConv(
    base_path="/discover/nobackup/projects/gmao/merra21c/TSE_staging",
    experiment_id="e5303_m21c_jan18",
    time_tolerance=timedelta(hours=2)
)
conv_df = conv_ds(datetime(2022, 1, 1, 0), ["t", "u", "v"])
print(f"Conventional observations: {len(conv_df)} rows")
print(conv_df.head())

# Satellite observations - use MERRA21cObsSat
sat_ds = MERRA21cObsSat(
    base_path="/discover/nobackup/projects/gmao/merra21c/TSE_staging",
    experiment_id="e5303_m21c_jan18",
    time_tolerance=timedelta(hours=2),
    satellites=["n20", "npp"]
)
sat_df = sat_ds(datetime(2022, 1, 1, 0), ["atms", "amsua"])
print(f"Satellite observations: {len(sat_df)} rows")
print(sat_df.head())

# Satellite observations - use MERRA21cObsSat
sat_ds = MERRA21cObsSat(
    base_path="/discover/nobackup/projects/gmao/merra21c/TSE_staging",
    experiment_id="e5303_m21c_jan18",
    time_tolerance=timedelta(hours=2),
    satellites=["metop-b", "npp"]
)
sat_df = sat_ds(datetime(2022, 1, 1, 0), ["amsua"])
print(f"Satellite observations: {len(sat_df)} rows")
print(sat_df.head())