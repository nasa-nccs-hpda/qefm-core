import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
import pandas as pd
from pathlib import Path
import argparse

def load_input_upper(input_data_dir: str, input_file: str, tidx:int = 0) -> np.ndarray:
    data_root = Path(input_data_dir)
    if not (data_root / input_file).exists():
        raise FileNotFoundError(f'{input_file} not found in {input_data_dir}') 
    ds = xr.open_dataset(data_root / input_file)
    variables = ['z', 'q', 't', 'u', 'v']
    data = np.stack([ds[var].isel(valid_time=tidx).values for var in variables], axis=0)
    return data

def load_input_surface(input_data_dir: str, input_file: str, tidx:int = 0) -> np.ndarray:
    data_root = Path(input_data_dir)
    if not (data_root / input_file).exists():
        raise FileNotFoundError(f'{input_file} not found in {input_data_dir}') 
    ds = xr.open_dataset(data_root / input_file)
    variables = ['msl', 'u10', 'v10', 't2m']
    data = np.stack([ds[var].isel(valid_time=tidx).values for var in variables], axis=0)
    return data

def surface_to_ds(arr: np.ndarray, time_value: np.datetime64) -> xr.Dataset:
    # Add time dimension
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # Save the surface numpy arrays to xarray dataset
    ds = xr.Dataset(
        {
            'msl': (('time', 'lat', 'lon'), arr[0,0,...]),
            'u10': (('time', 'lat', 'lon'), arr[0,1,...]),
            'v10': (('time', 'lat', 'lon'), arr[0,2,...]),
            't2m': (('time', 'lat', 'lon'), arr[0,3,...]),
        },
        coords={
            'time': [time_value],
            'lat': np.linspace(90, -90, 721),
            'lon': np.linspace(0, 359.75, 1440),
        },
    )
    return ds 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Pangu-Weather inference')
    parser.add_argument('--input_data_dir', type=str, default='input_data', help='The directory of your input data')
    parser.add_argument('--output_data_dir', type=str, default='output_data', help='The directory of your output data')
    parser.add_argument('--ckpt_path', type=str, default='./', help='The path to the pre-trained model')
    parser.add_argument('--start_time', type=str, default='2024-12-01T00:00:00', help='The start time of the inference')
    parser.add_argument('--time_steps', type=int, default=2, help='The number of time steps to run the inference')
    args = parser.parse_args()

    # The directory of your input and output data
    input_data_dir = args.input_data_dir
    output_data_dir = args.output_data_dir
    
    # The start time of the inference  
    # & the number of inference steps
    start_time = args.start_time
    nsteps = args.time_steps

    # Load the pre-trained Pangu-Weather models
    pangu_weather_24 = Path(args.ckpt_path) / 'pangu_weather_24.onnx'
    pangu_weather_6 = Path(args.ckpt_path) / 'pangu_weather_6.onnx'
    assert pangu_weather_24.exists() and pangu_weather_6.exists()

    model_24 = onnx.load(pangu_weather_24)
    model_6 = onnx.load(pangu_weather_6)
  
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session_24 = ort.InferenceSession(str(pangu_weather_24), sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
    ort_session_6 = ort.InferenceSession(str(pangu_weather_6), sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

    # Load the upper-air variables
    input = load_input_upper(input_data_dir, f'{start_time[:10]}-atmospheric.nc', tidx=0)

    # Load the surface variables
    input_surface = load_input_surface(input_data_dir, f'{start_time[:10]}-surface-level.nc', tidx=0)

    # Run the inference session
    
    time_values = pd.date_range(start=start_time, periods=nsteps, freq='6H')
    input_24, input_surface_24 = input, input_surface
    
    datasets_surface = []
    for i in range(nsteps):
      if (i+1) % 4 == 0:
        output, output_surface = ort_session_24.run(None, {'input':input_24, 'input_surface':input_surface_24})
        ds_tmp = surface_to_ds(output_surface, time_values[i+1])
        datasets_surface.append(ds_tmp)
        input_24, input_surface_24 = output, output_surface
        np.save(os.path.join(output_data_dir, f'output_upper_tidx_{i+1:02d}'), output)
        np.save(os.path.join(output_data_dir, f'output_surface_tidx_{i+1:02d}'), output_surface)
      else:
        output, output_surface = ort_session_6.run(None, {'input':input, 'input_surface':input_surface})
        ds_tmp = surface_to_ds(output_surface, time_values[i+1])
        print(ds_tmp)
        datasets_surface.append(ds_tmp)
        exit()
        input, input_surface = output, output_surface
        # Your can save the results here
        # Save the results
        #np.save(os.path.join(output_data_dir, f'output_upper_tidx_{i+1:02d}'), output)
        #np.save(os.path.join(output_data_dir, f'output_surface_tidx_{i+1:02d}'), output_surface)
