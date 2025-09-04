fm="FMGraphCast"
# path="/discover/nobackup/jli30/QEFM/qefm-core"

# cd "$path"/qefm/models/src/"$fm"

#cd "/explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast"
current_dir=$(pwd)
# if [[ ! -z "${PYTHONPATH}" ]]; then
#     echo "PYTHONPATH: "$PYTHONPATH""
# fi

module load anaconda
conda activate graphcast-env
YYYY=2024
MM=12
 
for DD in {08..08}; do
    cmd="python ./_graphcast_cf_init.py --year "$YYYY" --month "$MM" --day "$DD" --indir /explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/rollout_outputs/v20240903 --outdir /explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/rollout_outputs/v20240903/cf "
    #cmd="singularity exec --nv -B /explore/nobackup/projects/ilab /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_graphcast_cf_init.py --year "$YYYY" --month "$MM" --day "$DD" --indir /discover/nobackup/projects/QEFM/data/FMGraphCast/rollout_outputs/v20250815 --outdir /discover/nobackup/projects/QEFM/data/FMGraphCast/rollout_outputs/v20250815/cf --tsteps 41 "
    echo $fm: $cmd
    $cmd    
    cmd="python ./_graphcast_cf.py --year "$YYYY" --month "$MM" --day "$DD" --indir /explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/rollout_outputs/v20240903 --outdir /explore/nobackup/projects/ilab/projects/QEFM/data/FMGraphCast/rollout_outputs/v20240903/cf "
    #cmd="singularity exec --nv -B /explore/nobackup/projects/ilab /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_graphcast_cf.py --year "$YYYY" --month "$MM" --day "$DD"  --indir /discover/nobackup/projects/QEFM/data/FMGraphCast/rollout_outputs/v20250815 --outdir /discover/nobackup/projects/QEFM/data/FMGraphCast/rollout_outputs/v20250815/cf --tsteps 41"
    echo $fm: $cmd
    $cmd
done
