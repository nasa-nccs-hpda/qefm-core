fm="GraphCast"

cd "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast"
# cd "$path"/qefm/models/src/FMGraphCast
# current_dir=$(pwd)
# if [[ ! -z "${PYTHONPATH}" ]]; then
#     echo "PYTHONPATH: "$PYTHONPATH""
# fi

module load singularity

YYYY=$1
MM=$2
D1=$3
D2=$4
freq=$5
nsteps=$6
nlevs=$7
outdir=$8

vers=""
# outdir="/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/6h/Y2024/var"

for j in $(seq $D1 $D2); do
    printf -v DD "%02d" "$j"
    filename="$outdir"/"$vers"/graphcast-dataset-source-era5_date-"$YYYY"-"$MM"-"$DD"_res-0.25_levels-"$nlevs"_freq-"$freq"_steps-"$nsteps".nc""
    if [ -e $filename ]; then
        echo "$filename exists."
    else
        echo "$filename does not exist so we'll create it."

        cmd="python _graphcast_input_layers.py --outdir "$outdir"/"$vers" --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq"h --nsteps "$nsteps" --levs "$nlevs" "
        #cmd="singularity exec --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/people/gtamkin/.nccstmp /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  
        #python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_graphcast_input_layers.py --outdir "$outdir" --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq"h --nsteps "$nsteps" --levs "$nlevs" "
        echo $fm: $cmd
        $cmd
    fi
done
