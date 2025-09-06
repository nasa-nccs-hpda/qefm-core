import optax
import xarray
from typing import Any, Dict, Tuple, Iterator, List
from graphcast import data_utils
from graphcast import graphcast

import dataclasses
import os
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

if __name__ == "__main__":
    #from graphcast import config as config_lib
    #from graphcast import data_utils
    import glob

    # Example usage
    #config = config_lib.get_config()
    task_config = graphcast.TASK_13_PRECIP_OUT

    # Assuming you have a list of file paths
    input_dir = "/discover/nobackup/projects/QEFM/data/FMGenCast/6hr/samples/graph/"
    file_list = glob.glob(os.path.join(input_dir, "graph*2022*steps-4.nc"))  # Update with your data path

    batch_size = 4
    target_lead_times = slice("6h", "12h")

    data_loader = batch_data_loader(file_list, task_config, batch_size, target_lead_times)

    for inputs, targets, forcings in data_loader:
        print("Batch Inputs: ", inputs.dims.mapping)
        print("Batch Targets:", targets.dims.mapping)
        print("Batch Forcings:", forcings.dims.mapping)
        break  # Remove this break to process all batches