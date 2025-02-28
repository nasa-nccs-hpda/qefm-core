import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
from pathlib import Path

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

# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'
model_24 = onnx.load('pangu_weather_24.onnx')
model_6 = onnx.load('pangu_weather_6.onnx')

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
ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
ort_session_6 = ort.InferenceSession('pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
input2 = load_input_upper(input_data_dir, '2024-12-01-atmospheric.nc', tidx=0)
assert np.allclose(input, input2)

# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
input_surface2 = load_input_surface(input_data_dir, '2024-12-01-surface-level.nc', tidx=0)
assert np.allclose(input_surface, input_surface2)

exit()
# Run the inference session
input_24, input_surface_24 = input, input_surface
for i in range(60):
  if (i+1) % 4 == 0:
    output, output_surface = ort_session_24.run(None, {'input':input_24, 'input_surface':input_surface_24})
    input_24, input_surface_24 = output, output_surface
    np.save(os.path.join(output_data_dir, f'output_upper_tidx_{i+1:02d}'), output)
    np.save(os.path.join(output_data_dir, f'output_surface_tidx_{i+1:02d}'), output_surface)
  else:
    output, output_surface = ort_session_6.run(None, {'input':input, 'input_surface':input_surface})
    input, input_surface = output, output_surface
    # Your can save the results here
    # Save the results
    np.save(os.path.join(output_data_dir, f'output_upper_tidx_{i+1:02d}'), output)
    np.save(os.path.join(output_data_dir, f'output_surface_tidx_{i+1:02d}'), output_surface)
