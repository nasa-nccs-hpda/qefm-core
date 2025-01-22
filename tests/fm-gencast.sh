fm="GraphCast"
container="$1"/../containers/"$2"

cd "$1"/qefm/models/src/FMGenCast
current_dir=$(pwd)
export PYTHONPATH="$1"/qefm/models/src/FMGenCast/graphcast
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
cmd="time singularity exec --nv -B "$1"/qefm  "$1"/../containers/"$2" python -u -m torch.distributed.run "$1"/qefm/models/src/FMGenCast/fm_gencast.py"
echo $fm: $cmd
$cmd

