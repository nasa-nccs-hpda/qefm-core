#!/bin/csh -e
#SBATCH --time=01:00:00
#SBATCH --job-name=ana_default
#SBATCH --output=/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_082906/logs/ana_default.%j.out
#SBATCH --constraint=mil
#SBATCH --account=qefm
#SBATCH --partition=packable
#SBATCH --qos=qkst_pk
#SBATCH --mem=16G

source /usr/share/lmod/lmod/init/csh
module load python/GEOSpyD
cd /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats

python -u stats.py --config /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_082906/jobs/stats_GraphCast_ERA5_ERA5.yaml --ana --collection default --info_dir stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_082906
