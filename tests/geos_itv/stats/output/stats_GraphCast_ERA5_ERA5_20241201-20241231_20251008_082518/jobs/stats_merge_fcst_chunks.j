#!/bin/csh -e

#SBATCH --time=00:10:00
#SBATCH --job-name=merge_fcst_chunks
#SBATCH --output=/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_082518/logs/merge_fcst_chunks.%j.out
#SBATCH --constraint=mil
#SBATCH --account=gtamkin
#SBATCH --partition=packable
#SBATCH --qos=qkst_pk
#SBATCH --mem=8G
#SBATCH --dependency=afterok:FCST_CHUNK_DEP_PLACEHOLDER

source /usr/share/lmod/lmod/init/csh
module load nco

cd /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats

# Find all forecast chunk files (sorted)
set chunk_files = `ls /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_082518/tmp/fcst_chunk_*.nc4 | sort`
echo "Found $#chunk_files forecast chunk files to merge"

if ($#chunk_files == 0) then
    echo "ERROR: No forecast chunk files found! Exiting."
    exit 1
endif

# Define output file path - save to regular output if no collection merge needed
if (0) then
    set output_file = "/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_082518/tmp/fcst_GraphCast_20241201-20241231_len10d_int12h_spc1d_91x144.nc4"
else
    # No collection merge needed - save directly to output directory
    set output_file = "/gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/fcst_GraphCast_20241201-20241231_len10d_int12h_spc1d_91x144.nc4"
endif
echo "Output file: $output_file"

# Use ncrcat to concatenate along record dimension (init_date)
echo "Merging files with ncrcat..."
ncrcat $chunk_files $output_file

# Cleanup on success
if ($status == 0) then
    echo "Merge successful, cleaning up temp files"
    rm /gpfsm/dnb06/projects/p276/qefm-core/tests/geos_itv/stats/output/stats_GraphCast_ERA5_ERA5_20241201-20241231_20251008_082518/tmp/fcst_chunk_*.nc4
else
    echo "Merge failed, keeping temp files for debugging"
    exit 1
endif

echo "Merge complete for forecast chunks: $output_file" 
