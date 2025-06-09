fm="GraphCast"
container="$1"/../containers/"$2"

cd "$1"/qefm/models/src/FMGraphCast
current_dir=$(pwd)
export PYTHONPATH="$1"/qefm/models/src/FMGraphCast/graphcast
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
username=$(whoami)
cmd="time singularity exec --nv -B /home/"$username","$1"/qefm,/discover/nobackup/"$username"  "$1"/../containers/"$2" python "$1"/qefm/models/src/FMGraphCast/fm_graphcast.py"
echo $fm: $cmd
$cmd

