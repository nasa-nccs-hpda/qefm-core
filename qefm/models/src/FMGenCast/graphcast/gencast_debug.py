from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning
import os
import dataclasses
import optax
from typing import Any, Dict, Tuple, Iterator, List

import haiku as hk
import jax
import numpy as np
import xarray
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__name__))
print("script_dir:\n", script_dir, "\n")

## Load model from ckpt
params_file_value = "GenCast 1p0deg Mini <2019.npz"
#relative_params_file = '../../../checkpoints/gencast/gencast-params-GenCast_1p0deg_Mini_<2019.npz'
relative_params_file = '/explore/nobackup/people/jli30/workspace/qefm-core/qefm/models/checkpoints/gencast/gencast-params-GenCast_1p0deg_Mini_<2019.npz'
#relative_params_file = '/explore/nobackup/people/jli30/workspace/qefm-core/qefm/models/checkpoints/gencast/gencast-params-GenCast_0p25deg<2019.npz'
absolute_path = os.path.join(script_dir, relative_params_file)
print("absolute_path:\n", absolute_path, "\n")
params_file = absolute_path
with open(params_file, "rb") as f:
    print(params_file)
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
print("Task config:\n", task_config, "\n")
print("Sampler config:\n", sampler_config, "\n")
print("Noise config:\n", noise_config, "\n")
print("Noise encoder config:\n", noise_encoder_config, "\n")
print("Denoiser architecture config:\n", denoiser_architecture_config, "\n")

# ## Load the example data
def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))


def data_valid_for_model(file_name: str, params_file_name: str):
  """Check data type and resolution matches."""
  data_file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  data_res = data_file_parts["res"].replace(".", "p")
  res_matches = data_res in params_file_name.lower()
  source_matches = "Operational" in params_file_name
  if data_file_parts["source"] == "era5":
    source_matches = not source_matches
  return res_matches and source_matches

# @title Load weather data
# dataset_file_value= "source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc"
dataset_file_value = "gencast-dataset-source-era5_date-2024-12-10_res-1.0_levels-13_steps-10.nc"
dataset_dir = "/explore/nobackup/people/jli30/workspace/qefm-core/qefm/models/checkpoints/gencast"
#dataset_dir = "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/checkpoints/gencast" 
#dataset_dir = "/discover/nobackup/projects/QEFM/data/FMGenCast/12hr/Y2024"
#dataset_file_value = "gencast-dataset-source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc" 
#dataset_file_value = "source-era5_date-2019-03-29_res-0.25_levels-13_steps-04.nc"
dataset_file = os.path.join(dataset_dir, dataset_file_value)
print("dataset_file_value:\n", dataset_file_value, "\n")
# with gcs_bucket.blob(dir_prefix + f"dataset/{dataset_file_value}").open("rb") as f:
with open(dataset_file, "rb") as f:
  example_batch = xarray.load_dataset(f).compute()
##example_batch = xarray.open_dataset(dataset_file)

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file_value.removesuffix(".nc")).items()]))

#print(example_batch['2m_temperature'].isel(time=0).squeeze().to_numpy())
##example_batch
# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", "12h"), # Only 1AR training.
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", f"{(example_batch.dims['time']-2)*12}h"), # All but 2 input frames.
    **dataclasses.asdict(task_config))
print("Train Inputs shape: ", train_inputs)
print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

# @title Load normalization data
relative_diffs_file = "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/checkpoints/gencast/gencast-stats-diffs_stddev_by_level.nc"
diffs_file = os.path.join(script_dir, relative_diffs_file)

relative_mean_file = "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/checkpoints/gencast/gencast-stats-mean_by_level.nc"
mean_file = os.path.join(script_dir, relative_mean_file)

relative_stddev_file = "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/checkpoints/gencast/gencast-stats-stddev_by_level.nc"
stddev_file = os.path.join(script_dir, relative_stddev_file)

relative_min_file = "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/checkpoints/gencast/gencast-stats-min_by_level.nc"
min_file = os.path.join(script_dir, relative_min_file)

with open(diffs_file, "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(mean_file, "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open(stddev_file, "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()
with open(min_file, "rb") as f:
    min_by_level = xarray.load_dataset(f).compute()

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

grads_fn_jitted = jax.jit(grads_fn)
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
loss_fn_jitted = jax.jit(
   lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0])

# loss, diagnostics = loss_fn_jitted(
#    jax.random.PRNGKey(0),
#    train_inputs,
#    train_targets,
#    train_forcings)
# print("Loss:", float(loss))


# loss, diagnostics, next_state, grads = grads_fn_jitted(
#    params=params,
#    state=state,
#    inputs=train_inputs,
#    targets=train_targets,
#    forcings=train_forcings)
# mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
# print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")

# Load training data
def extract_example(file_path, task_config, target_lead_times=slice("6h", "6h")) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """Extracts inputs, targets, and forcings from a single example file."""
    with open(file_path, "rb") as f:
        ds = xarray.load_dataset(f).compute()
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds,
        target_lead_times=target_lead_times,
#        input_duration='12h',
        **dataclasses.asdict(task_config)
    )
    print("Batched Inputs:  ", inputs.dims.mapping)
    return (inputs, targets, forcings)

def batch_data_loader(file_list: List[str], task_config, batch_size: int = 1, target_lead_times=slice("12h", "12h")) -> Iterator[Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]]:
    """Generator to yield batches of inputs, targets, and forcings."""
    batch = []
    for file in file_list:

        example = extract_example(file, task_config, target_lead_times)
        batch.append(example)

        if len(batch) == batch_size:
           
           yield collate_batch(batch)
           batch = []


def collate_batch(batch: List[Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]]) -> Tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """Collates a list of examples into a single batch."""
    inputs, targets, forcings = zip(*batch)
    inputs = xarray.concat(inputs, dim="batch")
    targets = xarray.concat(targets, dim="batch")
    forcings = xarray.concat(forcings, dim="batch")
    return inputs, targets, forcings


def write_checkpoint(
    path_scheme: str, 
    epoch_number: int, 
    params: dict[str, Any], 
    description: str = ckpt.description,
    license: str = ckpt.license,
    task_config = ckpt.task_config,
    denoiser_architecture_config = ckpt.denoiser_architecture_config,
    sampler_config = ckpt.sampler_config,
    noise_config = ckpt.noise_config,
    noise_encoder_config = ckpt.noise_encoder_config):
    checkpoint_filename = path_scheme.format(ep_number=epoch_number)
    with open(checkpoint_filename, 'wb') as cfile:
        checkpoint.dump(cfile, gencast.CheckPoint(params=params,
                                                task_config=task_config,
                                                denoiser_architecture_config=denoiser_architecture_config,
                                                sampler_config=sampler_config,
                                                noise_config=noise_config,
                                                noise_encoder_config=noise_encoder_config,
                                                description=f"Model checkpoint epoch {ep_number}",
                                                license=""))


# # setup optimiser
# lr = 1e-3
# optimizer = optax.adam(learning_rate=lr, b1=0.9, b2=0.999, eps=1e-8)
# opt_state = optimizer.init(params)

# updates, opt_state = optimizer.update(
#     grads, opt_state, params=params
# )
# params = optax.apply_updates(params, updates)

# @title Training loop
num_epochs = 1000
batch_size = 1

dataset_dir = Path("/explore/nobackup/projects/ilab/data/qefm/gencast/input/6hr/")
file_list = sorted(dataset_dir.glob("*date-2020*.nc"))
lr = 1e-3
optimizer = optax.adam(learning_rate=lr, b1=0.9, b2=0.999, eps=1e-8)
opt_state = optimizer.init(params)


# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    shuffled_files = np.random.permutation(file_list)

    for batched_inputs, batched_targets, batched_forcings in batch_data_loader(
        shuffled_files,
        task_config,
        batch_size=batch_size,
        target_lead_times=slice("12h", "12h")
    ):

        print(f"Processing batch with {len(batched_inputs['batch'])} examples")

        # drop batch dimension for land_sea_mask & geopotential_at_surface
        for var in ['land_sea_mask', 'geopotential_at_surface']:
            if var in batched_inputs:
                batched_inputs[var] = batched_inputs[var].isel(batch=0)
            if var in batched_targets:
                batched_targets[var] = batched_targets[var].isel(batch=0)
        # Ensure inputs, targets, and forcings are xarray datasets
        assert isinstance(batched_inputs, xarray.Dataset)
        assert isinstance(batched_targets, xarray.Dataset)
        assert isinstance(batched_forcings, xarray.Dataset)

        # Compute loss and gradients
        loss, diagnostics, next_state, grads = grads_fn_jitted(
            params, state, batched_inputs, batched_targets, batched_forcings
        )
        #mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])

        # Update model parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Update model state (optional: only if model uses state)
        state = next_state

    # Optional: print training progress
    print(f"Epoch {epoch}, Loss: {loss}")

    path_scheme = "/explore/nobackup/people/jli30/workspace/qefm-core/qefm/models/src/FMGenCast/graphcast/checkpoints/GenCast.1p0deg.epoch.{ep_number:05d}.npz"
    if (epoch > 0 and epoch % 2 == 0):
        write_checkpoint(path_scheme, epoch, params)
        exit() 
