#!/usr/bin/env python
# coding: utf-8

# > Copyright 2024 DeepMind Technologies Limited.
# >
# > Licensed under the Apache License, Version 2.0 (the "License");
# > you may not use this file except in compliance with the License.
# > You may obtain a copy of the License at
# >
# >      http://www.apache.org/licenses/LICENSE-2.0
# >
# > Unless required by applicable law or agreed to in writing, software
# > distributed under the License is distributed on an "AS-IS" BASIS,
# > WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# > See the License for the specific language governing permissions and
# > limitations under the License.

# # GenCast Mini Demo
# 
# This notebook demonstrates running `GenCast 1p0deg Mini <2019`.
# 
# `GenCast 1p0deg Mini <2019` is a GenCast model at 1deg resolution, with 13 pressure levels and a 4 times refined icosahedral mesh. It is trained on ERA5 data from 1979 to 2018, and can be causally evaluated on 2019 and later years.
# 
# While other GenCast models are [available](https://github.com/google-deepmind/graphcast/blob/main/README.md), this model has the smallest memory footprint of those provided and is the only one runnable with the freely provided TPUv2-8 configuration in Colab. You can select this configuration in `Runtime>Change Runtime Type`.
# 
# **N.B.** The performance of `GenCast 1p0deg Mini <2019` is reasonable but is not representative of the performance of the other GenCast models described in the [README](https://github.com/google-deepmind/graphcast/blob/main/README.md).
# 
# To run the other models using Google Cloud Compute, refer to [gencast_demo_cloud_vm.ipynb](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_demo_cloud_vm.ipynb).

# # Installation and Initialization

# In[ ]:


# @title Upgrade packages (kernel needs to be restarted after running this cell).

#%pip install -U importlib_metadata


# In[ ]:


# @title Pip install repo and dependencies

#%pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip


# In[ ]:

print("\n====================================")
print("NOTE: These need to be installed in the container:")
print("---> pip install tree_math")
print("---> pip install xarray_tensorstore")
print("====================================\n")
#%pip install tree_math


# In[ ]:


#%pip install xarray_tensorstore


# In[1]:


# @title Imports

import dataclasses
import datetime
import math
from google.cloud import storage
from typing import Optional
import haiku as hk
from IPython.display import HTML
from IPython import display
# import ipywidgets as widgets
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning



# In[2]:


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

  ani = animation.FuncAnimation(
      fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return HTML(ani.to_jshtml())


# # Load the Data and initialize the model

# In[3]:

import os
script_dir = os.path.dirname(os.path.abspath(__name__))
print("script_dir:\n", script_dir, "\n")
# @title Authenticate with Google Cloud Storage

# Gives you an authenticated client, in case you want to use a private bucket.
# gcs_client = storage.Client.create_anonymous_client()
# gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"


# ## Load the model params
# 
# Choose one of the two ways of getting model params:
# - **checkpoint**: You'll get sensible predictions, but are limited to the model architecture that it was trained with, which may not fit on your device.
# - **random**: You'll get random predictions, but you can change the model architecture and data resolution which may run faster or fit on your device.
# 
# 

# In[5]:


# @title Choose the model

# params_file_options = [
#     name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix+"params/"))
#     if (name := blob.name.removeprefix(dir_prefix+"params/"))]  # Drop empty string.

latent_value_options = [int(2**i) for i in range(4, 10)]
# random_latent_size = widgets.Dropdown(
#     options=latent_value_options, value=512,description="Latent size:")
# random_attention_type = widgets.Dropdown(
#     options=["splash_mha", "triblockdiag_mha", "mha"], value="splash_mha", description="Attention:")
# random_mesh_size = widgets.IntSlider(
#     value=4, min=4, max=6, description="Mesh size:")
# random_num_heads = widgets.Dropdown(
#     options=[int(2**i) for i in range(0, 3)], value=4,description="Num heads:")
# random_attention_k_hop = widgets.Dropdown(
#     options=[int(2**i) for i in range(2, 5)], value=16,description="Attn k hop:")
# random_resolution = widgets.Dropdown(
#     options=["1p0", "0p25"], value="1p0", description="Resolution:")
#

random_latent_size = 512
random_attention_type = "splash_mha"
random_mesh_size = 4
random_num_heads = 4
random_attention_k_hop = 16
random_resolution = "1p0"

def update_latent_options(*args):
  def _latent_valid_for_attn(attn, latent, heads):
    head_dim, rem = divmod(latent, heads)
    if rem != 0:
      return False
    # Required for splash attn.
    if head_dim % 128 != 0:
      return attn != "splash_mha"
    return True
  attn = random_attention_type.value
  heads = random_num_heads.value
  random_latent_size.options = [
      latent for latent in latent_value_options
      if _latent_valid_for_attn(attn, latent, heads)]

# Observe changes to only allow for valid combinations.
# random_attention_type.observe(update_latent_options, "value")
# random_latent_size.observe(update_latent_options, "value")
# random_num_heads.observe(update_latent_options, "value")

# params_file = widgets.Dropdown(
#     options=[f for f in params_file_options if "Mini" in f],
#     description="Params file:",
#     layout={"width": "max-content"})
#
# source_tab = widgets.Tab([
#     params_file,
#     widgets.VBox([
#         random_attention_type,
#         random_mesh_size,
#         random_num_heads,
#         random_latent_size,
#         random_attention_k_hop,
#         random_resolution
#     ]),
# ])
# source_tab.set_title(0, "Checkpoint")
# source_tab.set_title(1, "Random")
# widgets.VBox([
#     source_tab,
#     widgets.Label(value="Run the next cell to load the model. Rerunning this cell clears your selection.")
# ])


# In[6]:


# @title Load the model

# source = source_tab.get_title(source_tab.selected_index)
# GST
source = "Checkpoint"
if source == "Random":
  params = None  # Filled in below
  state = {}
  task_config = gencast.TASK
  # Use default values.
  sampler_config = gencast.SamplerConfig()
  noise_config = gencast.NoiseConfig()
  noise_encoder_config = denoiser.NoiseEncoderConfig()
  # Configure, otherwise use default values.
  denoiser_architecture_config = denoiser.DenoiserArchitectureConfig(
    sparse_transformer_config = denoiser.SparseTransformerConfig(
        attention_k_hop=random_attention_k_hop.value,
        attention_type=random_attention_type.value,
        d_model=random_latent_size.value,
        num_heads=random_num_heads.value
        ),
    mesh_size=random_mesh_size.value,
    latent_size=random_latent_size.value,
  )
else:
  assert source == "Checkpoint"
  params_file_value = "GenCast 1p0deg Mini <2019.npz"
  relative_params_file = '../../checkpoints/gencast/gencast-params-GenCast_1p0deg_Mini_<2019.npz'
  absolute_path = os.path.join(script_dir, relative_params_file)
  print("absolute_path:\n", absolute_path, "\n")
  params_file = absolute_path
  with open(params_file, "rb") as f:
      # with gcs_bucket.blob(dir_prefix + f"params/{params_file_value}").open("rb") as f:
    print(f)
    ckpt = checkpoint.load(f, gencast.CheckPoint)
  params = ckpt.params
  state = {}

  task_config = ckpt.task_config
  sampler_config = ckpt.sampler_config
  noise_config = ckpt.noise_config
  noise_encoder_config = ckpt.noise_encoder_config
  denoiser_architecture_config = ckpt.denoiser_architecture_config

  denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
  denoiser_architecture_config.sparse_transformer_config.mask_type = "full"

  print("Model description:\n", ckpt.description, "\n")
  print("Model license:\n", ckpt.license, "\n")


# ## Load the example data
# 
# Example ERA5 datasets are available at 0.25 degree and 1 degree resolution.
# 
# Example HRES-fc0 datasets are available at 0.25 degree resolution.
# 
# Some transformations were done from the base datasets:
# - We accumulated precipitation over 12 hours instead of the default 1 hour.
# - For HRES-fc0 sea surface temperature, we assigned NaNs to grid cells in which sea surface temperature was NaN in the ERA5 dataset (this remains fixed at all times).
# 
# The data resolution must match the model that is loaded. Since we are running GenCast Mini, this will be 1 degree.
# 
# 

# In[8]:


# @title Get and filter the list of available example datasets

# dataset_file_options = [
#     name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix + "dataset/"))
#     if (name := blob.name.removeprefix(dir_prefix+"dataset/"))]  # Drop empty string.

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))


def data_valid_for_model(file_name: str, params_file_name: str):
  """Check data type and resolution matches."""
  data_file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  data_res = data_file_parts["res"].replace(".", "p")
  if source == "Random":
    return random_resolution.value == data_res
  res_matches = data_res in params_file_name.lower()
  source_matches = "Operational" in params_file_name
  if data_file_parts["source"] == "era5":
    source_matches = not source_matches
  return res_matches and source_matches

# dataset_file = widgets.Dropdown(
#     options=[
#         (", ".join([f"{k}: {v}" for k, v in parse_file_parts(option.removesuffix(".nc")).items()]), option)
#         for option in dataset_file_options
#         if data_valid_for_model(option, params_file.value)
#     ],
#     description="Dataset file:",
#     layout={"width": "max-content"})
# widgets.VBox([
#     dataset_file,
#     widgets.Label(value="Run the next cell to load the dataset. Rerunning this cell clears your selection and refilters the datasets that match your model.")
# ])


# In[9]:


# @title Load weather data
# dataset_file_value= "source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc"
dataset_file_value = "../../checkpoints/gencast/gencast-dataset-source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc"
dataset_file = os.path.join(script_dir, dataset_file_value)
print("dataset_file_value:\n", dataset_file_value, "\n")
# with gcs_bucket.blob(dir_prefix + f"dataset/{dataset_file_value}").open("rb") as f:
with open(dataset_file, "rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file_value.removesuffix(".nc")).items()]))

example_batch


# In[10]:


# @title Choose data to plot
#
# plot_example_variable = widgets.Dropdown(
#     options=example_batch.data_vars.keys(),
#     value="2m_temperature",
#     description="Variable")
# plot_example_level = widgets.Dropdown(
#     options=example_batch.coords["level"].values,
#     value=500,
#     description="Level")
# plot_example_robust = widgets.Checkbox(value=True, description="Robust")
# plot_example_max_steps = widgets.IntSlider(
#     min=1, max=example_batch.dims["time"], value=example_batch.dims["time"],
#     description="Max steps")
#
# widgets.VBox([
#     plot_example_variable,
#     plot_example_level,
#     plot_example_robust,
#     plot_example_max_steps,
#     widgets.Label(value="Run the next cell to plot the data. Rerunning this cell clears your selection.")
# ])


# In[11]:


# @title Plot example data

plot_size = 7
#
# data = {
#     " ": scale(select(example_batch, plot_example_variable.value, plot_example_level.value, plot_example_max_steps.value),
#               robust=plot_example_robust.value),
# }
# fig_title = plot_example_variable.value
# if "level" in example_batch[plot_example_variable.value].coords:
#   fig_title += f" at {plot_example_level.value} hPa"
#
# plot_data(data, fig_title, plot_size, plot_example_robust.value)


# In[12]:


# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", "12h"), # Only 1AR training.
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", f"{(example_batch.dims['time']-2)*12}h"), # All but 2 input frames.
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)


# In[13]:


# @title Load normalization data
relative_diffs_file = "../../checkpoints/gencast/gencast-stats-diffs_stddev_by_level.nc"
diffs_file = os.path.join(script_dir, relative_diffs_file)
print("diffs_file: ", str(diffs_file))

relative_mean_file = "../../checkpoints/gencast/gencast-stats-mean_by_level.nc"
mean_file = os.path.join(script_dir, relative_mean_file)
print("mean_file: ", str(mean_file))

relative_stddev_file = "../../checkpoints/gencast/gencast-stats-stddev_by_level.nc"
stddev_file = os.path.join(script_dir, relative_stddev_file)
print("stddev_file: ", str(stddev_file))

relative_min_file = "../../checkpoints/gencast/gencast-stats-min_by_level.nc"
min_file = os.path.join(script_dir, relative_min_file)
print("min_file: ", str(min_file))

with open(diffs_file, "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(mean_file, "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open(stddev_file, "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()
with open(min_file, "rb") as f:
    min_by_level = xarray.load_dataset(f).compute()

# with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
#   diffs_stddev_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
#   mean_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
#   stddev_by_level = xarray.load_dataset(f).compute()
# with gcs_bucket.blob(dir_prefix+"stats/min_by_level.nc").open("rb") as f:
#   min_by_level = xarray.load_dataset(f).compute()


# In[ ]:


# config.update("jax_platform_name", "cpu")

# Set the environment variable
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=24'
# # Verify it's set
# print(f"XLA_FLAGS: {os.getenv('XLA_FLAGS')}")
#
# print(jax.devices())
# jax.local_device_count(backend='cpu')


# In[14]:


# @title Build jitted functions, and possibly initialize random weights


def construct_wrapped_gencast():
  """Constructs and wraps the GenCast Predictor."""
  predictor = gencast.GenCast(
      sampler_config=sampler_config,
      task_config=task_config,
      denoiser_architecture_config=denoiser_architecture_config,
      noise_config=noise_config,
      noise_encoder_config=noise_encoder_config,
  )

  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level,
  )

  predictor = nan_cleaning.NaNCleaner(
      predictor=predictor,
      reintroduce_nans=True,
      fill_value=min_by_level,
      var_to_clean='sea_surface_temperature',
  )

  return predictor


@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
  predictor = construct_wrapped_gencast()
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(inputs, targets, forcings):
  predictor = construct_wrapped_gencast()
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics),
  )


def grads_fn(params, state, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), i, t, f
    )
    return loss, (diagnostics, next_state)

  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True
  )(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads


if params is None:
  init_jitted = jax.jit(loss_fn.init)
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets=train_targets,
      forcings=train_forcings,
  )


# loss_fn_jitted = jax.jit(
#     lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
# )

# loss_fn_jitted = jax.jit(
#     lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
# , backend='cpu')
# grads_fn_jitted = jax.jit(grads_fn, backend='cpu')
# run_forward_jitted = jax.jit(
#     lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
# , backend='cpu')
# # We also produce a pmapped version for running in parallel.
# run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample", backend='cpu')

# run_forward_jitted = jax.jit(
#     lambda rng, inputs, targets_template, forcings: run_forward.apply(params, state, rng, inputs, targets_template, forcings)[0]
# )
#GST orig below
grads_fn_jitted = jax.jit(grads_fn)
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
# We also produce a pmapped version for running in parallel.
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")


# # Run the model
# 
# The `chunked_prediction_generator_multiple_runs` iterates over forecast steps, where the 1 step forecast is jitted and samples are pmapped across the chips.
# This allows us to make efficient use of all devices and parallelise generating an ensemble across them. We then combine the chunks at the end to form our final forecast.
# 
# Note that the `Autoregressive rollout` cell will take longer than the standard inference time to run when executed for the first time, as this will include code compilation time. This cost does not increase with the number of devices, it is a fixed-cost one time operation whose result can be reused across any number of devices.

# In[15]:


# The number of ensemble members should be a multiple of the number of devices.
print(f"Number of local devices {len(jax.local_devices())}")


# In[16]:


# @title Autoregressive rollout (loop in python)

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

num_ensemble_members = 8 # @param int
rng = jax.random.PRNGKey(0)
# We fold-in the ensemble member, this way the first N members should always
# match across different runs which use take the same inputs, regardless of
# total ensemble size.
rngs = np.stack(
    [jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0)

chunks = []
# #chex.fake_pmap
# for chunk in rollout.chunked_prediction_generator_multiple_runs(
#     # Use pmapped version to parallelise across devices.
#     # predictor_fn=run_forward_pmap,
#     predictor_fn=run_forward_jitted,
#     rngs=rngs,
#     inputs=eval_inputs,
#     targets_template=eval_targets * np.nan,
#     forcings=eval_forcings,
#     num_steps_per_chunk = 1,
#     num_samples = num_ensemble_members,
#     # pmap_devices=jax.local_devices()
#     ):
#     chunks.append(chunk)
# predictions = xarray.combine_by_coords(chunks)

# GST
for chunk in rollout.chunked_prediction_generator_multiple_runs(
    # Use pmapped version to parallelise across devices.
    predictor_fn=run_forward_pmap,
    rngs=rngs,
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
    num_steps_per_chunk = 1,
    num_samples = num_ensemble_members,
    pmap_devices=jax.local_devices()
    ):
    chunks.append(chunk)
predictions = xarray.combine_by_coords(chunks)


# In[17]:


# @title Choose predictions to plot

# plot_pred_variable = widgets.Dropdown(
#     options=predictions.data_vars.keys(),
#     value="2m_temperature",
#     description="Variable")
# plot_pred_level = widgets.Dropdown(
#     options=predictions.coords["level"].values,
#     value=500,
#     description="Level")
# plot_pred_robust = widgets.Checkbox(value=True, description="Robust")
# plot_pred_max_steps = widgets.IntSlider(
#     min=1,
#     max=predictions.dims["time"],
#     value=predictions.dims["time"],
#     description="Max steps")
# plot_pred_samples = widgets.IntSlider(
#     min=1,
#     max=num_ensemble_members,
#     value=num_ensemble_members,
#     description="Samples")
#
# widgets.VBox([
#     plot_pred_variable,
#     plot_pred_level,
#     plot_pred_robust,
#     plot_pred_max_steps,
#     plot_pred_samples,
#     widgets.Label(value="Run the next cell to plot the predictions. Rerunning this cell clears your selection.")
# ])


# In[18]:


# @title Plot prediction samples and diffs

# plot_size = 5
# plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps.value)
#
# fig_title = plot_pred_variable.value
# if "level" in predictions[plot_pred_variable.value].coords:
#   fig_title += f" at {plot_pred_level.value} hPa"
#
# for sample_idx in range(plot_pred_samples.value):
#   data = {
#       "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
#       "Predictions": scale(select(predictions.isel(sample=sample_idx), plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
#       "Diff": scale((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps) -
#                           select(predictions.isel(sample=sample_idx), plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
#                         robust=plot_pred_robust.value, center=0),
#   }
#   display.display(plot_data(data, fig_title + f", Sample {sample_idx}", plot_size, plot_pred_robust.value))


# In[19]:


# @title Plot ensemble mean and CRPS

def crps(targets, predictions, bias_corrected = True):
  if predictions.sizes.get("sample", 1) < 2:
    raise ValueError(
        "predictions must have dim 'sample' with size at least 2.")
  sum_dims = ["sample", "sample2"]
  preds2 = predictions.rename({"sample": "sample2"})
  num_samps = predictions.sizes["sample"]
  num_samps2 = (num_samps - 1) if bias_corrected else num_samps
  mean_abs_diff = np.abs(
      predictions - preds2).sum(
          dim=sum_dims, skipna=False) / (num_samps * num_samps2)
  mean_abs_err = np.abs(targets - predictions).sum(dim="sample", skipna=False) / num_samps
  return mean_abs_err - 0.5 * mean_abs_diff


# plot_size = 5
# plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps.value)
#
# fig_title = plot_pred_variable.value
# if "level" in predictions[plot_pred_variable.value].coords:
#   fig_title += f" at {plot_pred_level.value} hPa"
#
# data = {
#     "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
#     "Ensemble Mean": scale(select(predictions.mean(dim=["sample"]), plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
#     "Ensemble CRPS": scale(crps((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
#                         select(predictions, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
#                       robust=plot_pred_robust.value, center=0),
# }
# display.display(plot_data(data, fig_title, plot_size, plot_pred_robust.value))


# # Train the model
# 
# The following operations requires larger amounts of memory than running inference.
# 
# The first time executing the cell takes more time, as it includes the time to jit the function.

# In[21]:


loss_fn_jitted = jax.jit(
    lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
)


# In[22]:


# @title Loss computation
loss, diagnostics = loss_fn_jitted(
    jax.random.PRNGKey(0),
    train_inputs,
    train_targets,
    train_forcings)
print("Loss:", float(loss))


# In[23]:


# @title Gradient computation
loss, diagnostics, next_state, grads = grads_fn_jitted(
    params=params,
    state=state,
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings)
mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")


# In[ ]:




