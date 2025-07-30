fm="GraphCast"
path="/discover/nobackup/jli30/QEFM/qefm-core"
#path="/discover/nobackup/jli30/QEFM/qefm-core"

cd "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast"
# cd "$path"/qefm/models/src/FMGraphCast
# current_dir=$(pwd)
# if [[ ! -z "${PYTHONPATH}" ]]; then
#     echo "PYTHONPATH: "$PYTHONPATH""
# fi
YYYY=2024
MM=12
module load anaconda
conda activate graphcast-env 
for DD in {01..31}; do
    filename="/discover/nobackup/projects/QEFM/data/FMGraphCast/6h/Y2024/graphcast-dataset-source-era5_date-"$YYYY"-"$MM"-"$DD"_res-0.25_levels-37_freq-6h_steps-20.nc"
    if [ -e $filename ]; then
        echo "$filename exists."
    else
        echo "$filename does not exist, so we'll create it."

        cmd="time python _graphcast_input.py --outdir /discover/nobackup/projects/QEFM/data/FMGraphCast/6h/Y2024 --year "$YYYY" --month "$MM" --day "$DD" --freq 6h --nsteps 22"
        echo $fm: $cmd
        $cmd
    fi
done
