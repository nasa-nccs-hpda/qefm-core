import cdsapi
import json
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='era5_download.log',  
    filemode='a'  
)

def get_ear5_vars(root_path="./"):
    """Get the list of variables in the ERA5 dataset."""
    surf_file = Path(root_path) / "era5_surf_variables.json"
    with open(surf_file, "r") as f:
        surf_vars = json.load(f)

    atmos_file = Path(root_path) / "era5_atmos_variables.json"
    with open(atmos_file, "r") as f:
        atmos_vars = json.load(f)
    return surf_vars, atmos_vars

def get_latest_date_in_month(root_path: str, year: str, month: str) -> datetime | None:
    """Find the latest date in a specific /year/month/day folder."""
    month_path = Path(root_path) / f"Y{year}" / f"M{month}"
    dates = []

    if not month_path.exists() or not month_path.is_dir():
        return f"No data for {year}-{month}"

    for day_dir in month_path.iterdir():
        if day_dir.is_dir() and day_dir.name.startswith("D"):  # Check if day folder
            try:
                day = day_dir.name[2:]  # Extract day part after 'D'
                date_str = f"{year}-{month}-{day}"
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_obj)
            except ValueError:
                pass  # Ignore invalid dates
    
    return max(dates) if dates else None

def check_avail_data(date_str: str) -> bool:
    """Check if data is available for a specific date."""
    c = cdsapi.Client()
    try:
        avail_data = c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': '2m_temperature',
                'year': date_str[:4],
                'month': date_str[4:6],
                'day': date_str[6:8],
                'time': '00:00',
                'format': 'netcdf'
            }
        )
        print(avail_data)
        return True
    except Exception as e:
        print(f"Failed to check available data for {date_str}: {e}")
        return False

def download_data(root_dir, date, surf_lst, atmos_lst):
    c = cdsapi.Client()
    current_date = date
    year = f"Y{current_date.year}"
    month = f"M{current_date.month:02d}"

    # get surface data for each time step
    # instantaneous surface data (as opposed to mean/max/min, column integrated, etc) 
          
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 data")
    parser.add_argument("--year", "-y", type=str, help="Year to download data for")
    parser.add_argument("--month", "-m", type=str, help="Month to download data for")
    args = parser.parse_args()
    
    if args.year and args.month:
        end_date = datetime(int(args.year), int(args.month), 25, 0, 0)
    else:
        end_date = (datetime.today().replace(hour=0.0) - timedelta(days=10)).date()
    logging.info(f"Downloading ERA5 data up to {end_date}")   
   
    era5_dir = Path("/css/era5")
    pred_dir = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/FMAurora")

    YYYY = end_date.strftime("%Y")
    MM = end_date.strftime("%m")
    DD = end_date.strftime("%d")

    last_date = get_latest_date_in_month(pred_dir, YYYY, MM)
    logging.info(f"Last date of ERA5 in {YYYY}-{MM} is {last_date}")

    date = last_date +timedelta(days=1)
    if date < end_date:
        if check_avail_data(date.strftime("%Y%m%d")):
            logging.info(f"Downloading ERA5 data from {date}")
            surf_lst, atmos_lst = get_ear5_vars(era5_dir)
            download_data(pred_dir, date, surf_lst=surf_lst, atmos_lst=atmos_lst)
            date += timedelta(days=1)

