import dataclasses
import functools

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import numpy as np
import xarray
from pathlib import Path
import argparse

print("Compute Graphcast prediction from subsetted ERA5 data:") 

parser = argparse.ArgumentParser(description="Compute Graphcast prediction from subsetted ERA5 data:")
#parser.add_argument("--infile", "-if", default=".graphcast-dataset-prediction-era5_date-2024-12-01_res-0.25_levels-37_freq-6h_steps-20.nc")
parser.add_argument("--indir", "-id", default="/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/6h/_Y2024", type=str, help="ERA5 subsetted source directory")
parser.add_argument("--outdir", "-o", default="/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/rollout_outputs/20240810", type=str, help="Graphcast Rolllout Output directory")
parser.add_argument("--year", "-y", default="24", type=str, help="Year of the data")
parser.add_argument("--month", "-m", default="12", type=str, help="Month of the data")
parser.add_argument("--day", "-d", default="01", type=str, help="Day of the data")
parser.add_argument("--freq", "-f", default="6", type=str, help="Frequency in hours")
parser.add_argument("--tsteps", "-t", default="1", type=str, help="# Training steps")
parser.add_argument("--esteps", "-e", default="20", type=str, help="# Eval steps")
parser.add_argument("--levs", "-l", default="37", type=str, help="Number of pressure levels")
parser.add_argument("--res", "-r", default="0.25", type=str, help="Data resolution")
parser.add_argument("--var", "-v", default="All", type=str, help="Variable of interest")

args = parser.parse_args()
date_str = f"{args.year}-{args.month}-{args.day}"
nsteps = int(args.esteps) 
start_time = f"{date_str}T00:00"
cfreq=f"{args.freq}"
levs=f"{args.levs}"
res=f"{args.res}"
var=f"{args.var}"
output_dir=Path(f"{args.outdir}")
print("arguments:", args._get_kwargs)
#infile=f"graphcast-dataset-source-era5_date-{date_str}_res-{res}_levels-{levs}_freq-{cfreq}h_steps-{nsteps}.nc"
#infile=f"graphcast-dataset-source-era5_date-{date_str}_res-{res}_levels-{levs}_freq-{cfreq}h_steps-20.nc"
infile=f"aggregated_graphcast-dataset-source-era5_date-{date_str}_var-ALL_res-{res}_levels-{levs}_freq-{cfreq}h_steps-42.nc"
predfile=f"graphcast-prediction-era5_date-{date_str}_res-{res}_levels-{levs}_freq-{cfreq}h_steps-{nsteps}.nc"
infile=f"{args.indir}/"+infile
predfile=f"{args.outdir}/"+predfile
print("infile:", infile)
print("predfile:", predfile)
input_source = "era5"


def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

import os
if os.path.exists(Path(predfile)):
  print("Prediction file already exists: ", predfile)
else:
  print("Prediction file doesn't exist: ", predfile)

  import os
  script_dir = os.path.dirname(os.path.abspath(__name__))
  relative_params_file = '../../checkpoints/graphcast/params_GraphCast_3d.npz'
  absolute_path = os.path.join(script_dir, relative_params_file)

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

  #date_str='2024-12-01'
  #relative_dataset_file = "../../checkpoints/graphcast/l37/graphcast-dataset-source-era5_date-"+str(date_str)+"_res-0.25_levels-37_steps-"+{str(nsteps-2)}.nc"
  relative_dataset= "../../checkpoints/graphcast/l37"

  #res="0.25"
  #nlev=37

  #dataset_file = os.path.join(script_dir, relative_dataset_file)
  dataset_file=Path(infile)
  print("dataset_file:\n", dataset_file, "\n")
  with open(dataset_file, "rb") as f:
      example_batch = xarray.load_dataset(f).compute()

  assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

  train_steps = int(args.tsteps)
  eval_steps = int(args.esteps)
  print("params_file: ", str(params_file))
  print("dataset_file: ", str(dataset_file))
  print("train_steps: ", str(train_steps))
  print("eval_steps: ", str(eval_steps))

  print('target_lead_times=slice("6h", f"{train_steps*6}h"')
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
  print("len(predictions)", len(predictions))
  #print("predictions:\n", predictions)
  out_dir = Path(args.outdir)
  out_file_value = f"graphcast-prediction-{input_source}_date-{date_str}_res-{res}_levels-{levs}_freq-{cfreq}h_steps-{eval_steps}.nc"
  out_file = os.path.join(out_dir, out_file_value)
  predictions.to_netcdf(out_file)
  days=(6*eval_steps)/24
  print("Predictions computed for "+str(days)+" days out_file:\n", out_file, "\n")
