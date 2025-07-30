fm="FMGraphCast"
# path="/discover/nobackup/jli30/QEFM/qefm-core"

# cd "$path"/qefm/models/src/"$fm"

cd "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast"
current_dir=$(pwd)
# if [[ ! -z "${PYTHONPATH}" ]]; then
#     echo "PYTHONPATH: "$PYTHONPATH""
# fi

module load singularity
YYYY=2024
MM=12
 
for DD in {01..31}; do
    filename="/explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/rollout_outputs/graphcast-prediction-era5_date-"$YYYY"-"$MM"-"$DD"_res-0.25_levels-37_eval_steps-20.nc"
    if [ -e $filename ]; then
        echo "$filename exists."
    else
        echo "$filename does not exist, so we'll create it."

        cmd="singularity exec --nv -B /explore/nobackup/projects/ilab /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_fm_graphcast.py --year "$YYYY" --month "$MM" --day "$DD" --freq 6 --esteps 20"
        echo $fm: $cmd
        $cmd
    fi
done
