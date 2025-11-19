#!/bin/csh -e
#SBATCH --time=00:30:00
#SBATCH --job-name=ana_slices
#SBATCH --output=/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251007_183252/logs/ana_slices.%j.out
#SBATCH --constraint=mil
#SBATCH --account=g0620
#SBATCH --partition=packable
#SBATCH --qos=qkst_pk
#SBATCH --mem=8G

source /usr/share/lmod/lmod/init/csh
module load python/GEOSpyD
cd /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats

python -u stats.py --config /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251007_183252/jobs/stats_GraphCast_ERA5_ERA5.yaml --ana --collection slices --info_dir stats_GraphCast_ERA5_ERA5_20241201-20241231_20251007_183252
