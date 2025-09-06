import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
# from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
#from IPython.display import HTML
#import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
import glob
import os

from ft_util import batch_data_loader

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

script_dir = os.path.dirname(os.path.abspath(__name__))
print("script_dir:\n", script_dir, "\n")
# relative_params_file = '../../checkpoints/graphcast/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"'
relative_params_file = '../../checkpoints/graphcast/params_GraphCast_small.npz'
absolute_path = os.path.join(script_dir, relative_params_file)
print("absolute_path:\n", absolute_path, "\n")

# params_file = "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
# with gcs_bucket.blob(f"params/{params_file}").open("rb") as f:
#     ckpt = checkpoint.load(f, graphcast.CheckPoint)

#params_file='../../checkpoints/graphcast/checkpoints/graphcast/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
#params_file='/explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/checkpoints/graphcast/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
params_file = absolute_path
with open(params_file, "rb") as f:
  ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

print("Model path:\n", params_file, "\n")

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model resolution:\n", model_config.resolution, "\n")
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

source = 'era5-mcdv3' #'era5-mcdv1' # 'era5'
#dataset_file = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
relative_dataset_file = f"../../checkpoints/graphcast/source-{source}_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
# relative_params_file = '../../checkpoints/graphcast/params_GraphCast_small.npz'
dataset_file = os.path.join(script_dir, relative_dataset_file)
print("dataset_file:\n", dataset_file, "\n")
# with gcs_bucket.blob(f"dataset/{dataset_file}").open("rb") as f:
with open(dataset_file, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

train_steps=1
eval_steps=4
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)
print("Eval eval_inputs.sizes[lon]: ", eval_inputs.sizes["lon"])

relative_stddev_file = "../../checkpoints/graphcast/stats_stddev_by_level.nc"
stddev_file = os.path.join(script_dir, relative_stddev_file)
print("stddev_file: ", str(stddev_file))

relative_mean_file = "../../checkpoints/graphcast/stats_mean_by_level.nc"
mean_file = os.path.join(script_dir, relative_mean_file)
print("mean_file: ", str(mean_file))

relative_diffs_file = "../../checkpoints/graphcast/stats_diffs_stddev_by_level.nc"
diffs_file = os.path.join(script_dir, relative_diffs_file)
print("diffs_file: ", str(diffs_file))

# with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
with open(diffs_file, "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
with open(mean_file, "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
with open(stddev_file, "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

# @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `FMGraphCast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(rng=jax.random.PRNGKey(0), **kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)

grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))

# @title Autoregressive rollout (loop in python)

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

# print("Inputs:  ", eval_inputs.dims.mapping)
# print("Targets: ", eval_targets.dims.mapping)
# print("Forcings:", eval_forcings.dims.mapping)

# predictions = rollout.chunked_prediction(
#     run_forward_jitted,
#     rng=jax.random.PRNGKey(0),
#     inputs=eval_inputs,
#     targets_template=eval_targets * np.nan,
#     forcings=eval_forcings)
# predictions
# print("predictions:\n", predictions)
# output_file = f"/discover/nobackup/jli30/mars/data/graph_output/fm_graphcast_jl_{source}_output.nc"
# predictions.to_netcdf(output_file)
# print(f"Saved predictions to {output_file}")

dataset_dir = "/discover/nobackup/projects/QEFM/data/FMGenCast/6hr/samples/graph/"
file_list = glob.glob(os.path.join(dataset_dir, "graph*2022*steps-4.nc"))
num_epochs = 1
batch_size = 4
target_lead_times = slice("6h", "12h")
for epoch in range(num_epochs):
   print(f"Epoch {epoch + 1}/{num_epochs}")

   shuffled_files = np.random.permutation(file_list)
   data_loader = batch_data_loader(file_list, task_config, batch_size, target_lead_times)
   for batched_inputs, batched_targets, batched_forcings in data_loader:
      print("batched_inputs.sizes:", batched_inputs.sizes)
      print("batched_targets.sizes:", batched_targets.sizes)
      print("batched_forcings.sizes:", batched_forcings.sizes)
      # loss, diagnostics, state, grads = grads_fn_jitted(
      #     params, model_config, task_config, 
      #     state, batched_inputs, batched_targets, batched_forcings)
      loss, diagnostics, next_state, grads = grads_fn_jitted(
         inputs=batched_inputs, targets=batched_targets, forcings=batched_forcings
      )
      print("Loss:", float(loss))
      break
   
      

         
     

   


     
