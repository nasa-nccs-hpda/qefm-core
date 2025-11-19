#!/bin/csh -e
#SBATCH --time=01:00:00
#SBATCH --job-name=fcst
#SBATCH --output=/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_091209/logs/fcst.%A_%a.out
#SBATCH --constraint=mil
#SBATCH --account=ilab
#SBATCH --partition=gpu_a100
#SBATCH --qos=8n_a100
#SBATCH --mem=16G
#SBATCH --array=1-7

source /usr/share/lmod/lmod/init/csh
module load python/GEOSpyD
cd /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats

set ID = ${SLURM_ARRAY_TASK_ID}
set dates_per_chunk = 5
set last_date_idx = 30

# Calculate start index in multiple steps
@ temp = $ID - 1
@ chunk_start_idx = $temp * $dates_per_chunk

# Calculate end index in multiple steps
@ chunk_end_idx = $chunk_start_idx + $dates_per_chunk
@ chunk_end_idx = $chunk_end_idx - 1

# Check bounds
if ($chunk_end_idx > $last_date_idx) then
    set chunk_end_idx = $last_date_idx
endif

echo "Processing chunk ${ID}: dates ${chunk_start_idx} - ${chunk_end_idx}"

python -u stats.py --config /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_091209/jobs/stats_GraphCast_ERA5_ERA5.yaml --fcst  --date_start_idx $chunk_start_idx --date_end_idx $chunk_end_idx --info_dir stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_091209
