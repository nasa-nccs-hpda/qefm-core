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
    cmd="singularity exec --nv -B /explore/nobackup/projects/ilab /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_graphcast_cf_init.py --year "$YYYY" --month "$MM" --day "$DD" "
    echo $fm: $cmd
    $cmd    
    cmd="singularity exec --nv -B /explore/nobackup/projects/ilab /explore/nobackup/projects/ilab/projects/QEFM/containers/qefm-core-gencast-20250511-sandbox/  python /explore/nobackup/projects/ilab/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/_graphcast_cf.py --year "$YYYY" --month "$MM" --day "$DD" "
    echo $fm: $cmd
    $cmd
done
