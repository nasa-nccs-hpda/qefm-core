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
from IPython.display import HTML
#import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray


def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

# @title Authenticate with Google Cloud Storage

# gcs_client = storage.Client.create_anonymous_client()
# gcs_bucket = gcs_client.get_bucket("dm_graphcast")

# @title Plotting functions

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
        plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
        origin="lower", cmap=cmap)
    plt.colorbar(
        mappable=im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.75,
        cmap=cmap,
        extend=("both" if robust else "neither"))
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

#   ani = animation.FuncAnimation(
#       fig=figure, func=update, frames=max_steps, interval=250)
#   plt.close(figure.number)
#   return HTML(ani.to_jshtml())

import os
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

#dataset_file = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
relative_dataset_file = "../../checkpoints/graphcast/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
# relative_params_file = '../../checkpoints/graphcast/params_GraphCast_small.npz'
dataset_file = os.path.join(script_dir, relative_dataset_file)
print("dataset_file:\n", dataset_file, "\n")
# with gcs_bucket.blob(f"dataset/{dataset_file}").open("rb") as f:
with open(dataset_file, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

train_steps = 1
eval_steps = 1
print("params_filex: ", str(params_file))
print("dataset_file: ", str(dataset_file))
print("train_steps: ", str(train_steps))
print("eval_steps: ", str(eval_steps))

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
  return lambda **kw: fn(**kw)[0]

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

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
predictions
print("predictions:\n", predictions)
