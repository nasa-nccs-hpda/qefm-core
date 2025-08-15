fm="GraphCast"

cd "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast"
# cd "$path"/qefm/models/src/FMGraphCast
# current_dir=$(pwd)
# if [[ ! -z "${PYTHONPATH}" ]]; then
#     echo "PYTHONPATH: "$PYTHONPATH""
# fi
YYYY=$1
MM=$2
freq=$3
nsteps=$4
nlevs=$5

# YYYY=2024
# MM=12
# nsteps=42
# freq=6h

#module load anaconda
#conda activate graphcast-env 
for DD in {01..01}; do
    filename="/discover/nobackup/projects/QEFM/data/FMGraphCast/6h/Y2024/graphcast-dataset-source-era5_date-"$YYYY"-"$MM"-"$DD"_res-0.25_levels-"$nlevs"_freq-"$freq"_steps-"$nsteps".nc"
    if [ -e $filename ]; then
        echo "$filename exists."
    else
        echo "$filename does not exist, so we'll create it."

        cmd="python _graphcast_input.py --outdir /discover/nobackup/projects/QEFM/data/FMGraphCast/6h/Y2024 --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq"h --nsteps "$nsteps" --levs "$nlevs" "
        echo $fm: $cmd
        $cmd
    fi
done
