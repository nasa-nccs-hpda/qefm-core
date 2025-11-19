#!/bin/csh -e

#SBATCH --time=00:10:00
#SBATCH --job-name=merge_ana
#SBATCH --output=/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_085538/logs/merge_ana.%j.out
#SBATCH --constraint=mil
#SBATCH --account=ilab
#SBATCH --partition=gpu_a100
#SBATCH --qos=debug
#SBATCH --mem=8G
#SBATCH --dependency=afterok:ANA_COLLECTION_DEP_PLACEHOLDER

source /usr/share/lmod/lmod/init/csh
module load python/GEOSpyD
cd /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats

python -u stats.py --config /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_085538/jobs/stats_GraphCast_ERA5_ERA5.yaml --merge_collections ana --info_dir stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_085538
