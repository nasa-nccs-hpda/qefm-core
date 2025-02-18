import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'geopotential', 'relative_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '50', '500', '850',
            '1000',
        ],
        'year': '2025',
        'month': '01',
        'day': [
            '01', '02', '03',
            '04', '05',
        ],
        'time': [
            '00:00', '06:00', '12:00',
            '18:00',
        ],
    },
    '/discover/nobackup/jli30/data/FNO/ERA5/jan_2025_01_05_pl.nc')
