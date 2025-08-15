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
freq=$3
nsteps=$4
 
for DD in {01..01}; do
    filename="/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/rollout_outputs/graphcast-prediction-era5_date-"$YYYY"-"$MM"-"$DD"_res-0.25_levels-37_eval_steps-"$nsteps".nc"
    if [ -e $filename ]; then
        echo "$filename exists."
    else
        echo "$filename does not exist, so we'll create it."

        cmd="singularity exec --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/people/gtamkin/.nccstmp /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_fm_graphcast.py --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq" --esteps "$nsteps" "
        #cmd="singularity exec --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/people/gtamkin /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_fm_graphcast.py --year "$YYYY" --month "$MM" --day "$DD" --freq "$freq" --esteps "$nsteps" "
        echo $fm: $cmd
        $cmd
    fi
done
