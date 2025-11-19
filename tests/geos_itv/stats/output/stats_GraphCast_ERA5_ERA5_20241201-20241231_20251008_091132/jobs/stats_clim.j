#!/bin/csh -e
#SBATCH --time=00:15:00
#SBATCH --job-name=clim
#SBATCH --output=/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_091132/logs/clim.%j.out
#SBATCH --constraint=mil
#SBATCH --account=ilab
#SBATCH --partition=gpu_a100
#SBATCH --qos=8n_a100
#SBATCH --mem=16G

source /usr/share/lmod/lmod/init/csh
module load python/GEOSpyD
cd /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats

python -u stats.py --config /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_091132/jobs/stats_GraphCast_ERA5_ERA5.yaml --clim --info_dir stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_091132
