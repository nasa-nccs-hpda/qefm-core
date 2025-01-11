fm="GraphCast"
container="$1"/../containers/"$2"
cd /explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/src/FMGraphCast
export PYTHONPATH=/explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/src/FMGraphCast/graphcast

cd "$1"/qefm/models/src/FMGraphCast
current_dir=$(pwd)
export PYTHONPATH="$1"/qefm/models/src/FMGraphCast/graphcast
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
cmd="time singularity exec --nv -B "$1"/qefm  "$1"/../containers/"$2" python -u -m torch.distributed.run "$1"/qefm/models/src/FMGraphCast/fm_graphcast.py"
echo $fm: $cmd
$cmd

