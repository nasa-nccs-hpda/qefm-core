# this script is used to upload the FMAurora rollout data to the HF database
# nasa-cisto-data-science-group/aurora_rollout_beta

from huggingface_hub import HfApi
from pathlib import Path

# Path to the directory containing the rollout data
root_dir = Path("/discover/nobackup/projects/QEFM/data/rollout_outputs/FMAurora")

# Define year and month
year = 2024
month = 12

api=HfApi()
local_folder = root_dir / f"Y{year}/M{month:02d}"
remote_folder = "FMAurora/rollout_data"

api.upload_folder(
    folder_path = local_folder,
    path_in_repo = remote_folder,
    repo_id = "nasa-cisto-data-science-group/aurora_rollout_beta",
    repo_type = "dataset",
)



                    

