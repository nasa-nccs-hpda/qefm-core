# Example:  gtamkin@discover11:/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast$ sh _fm_graphcast_era5_rollout_dynamic.sh 2024 12 6h 42

fm="FMGraphCast"
# path="/discover/nobackup/jli30/QEFM/qefm-core"

# cd "$path"/qefm/models/src/"$fm"

cd "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast"
current_dir=$(pwd)
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
indir=$8
outdir=$9

vers=""
ft="gdps"
for j in $(seq $D1 $D2); do
    printf -v DD "%02d" "$j"
    filename="$outdir"/"$vers"/aggregated_graphcast-dataset-source-era5_date-_"$YYYY"-"$MM"-"$DD"_var-ALL_res-0.25_levels-"$nlevs"_freq-"$freq"_steps-"$nsteps".nc" "

#    filename="/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/6h/Y2024/var/v20240901/graphcast-prediction-era5_date-"$YYYY"-"$MM"-"$DD"_res-0.25_levels-37_eval_steps-"$nsteps".nc"
    if [ -e $filename ]; then
        echo "$filename exists."
    else
        echo "$filename does not exist, so we'll create it."

        echo "Fine-tuning: $ft"
        if [[ "$ft" =~ "gdps" ]]; then
            cmd="singularity exec --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/people/gtamkin/.nccstmp /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_fm_graphcast_gdps.py --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq" --esteps "$nsteps" --indir "$indir" --outdir "$outdir" "
        else
#           cmd="singularity exec --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/people/gtamkin/.nccstmp /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-debian-all-aifs-20250609-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_fm_graphcast.py --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq" --esteps "$nsteps" --indir "$indir" --outdir "$outdir" "
            cmd="singularity exec --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/people/gtamkin/.nccstmp /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_fm_graphcast.py --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq" --esteps "$nsteps" --indir "$indir" --outdir "$outdir" "
            #cmd="singularity exec --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/people/gtamkin /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_fm_graphcast.py --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq" --esteps "$nsteps" "
        fi
        echo $fm: $cmd
        $cmd
    fi
done
