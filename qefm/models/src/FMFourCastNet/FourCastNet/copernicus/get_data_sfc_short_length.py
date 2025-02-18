import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'mean_sea_level_pressure', 'surface_pressure', 'total_column_water_vapour',
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
    '/discover/nobackup/jli30/data/FNO/ERA5/jan_2025_01_05_sfc.nc')


