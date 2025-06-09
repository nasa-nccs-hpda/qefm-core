fm="AIFS"
container="$1"/../containers/"$2"
cd "$1"/qefm/models/src/FMAifs
current_dir=$(pwd)
export PYTHONPATH="$1"/qefm/models/src/FMAifs
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
username=$(whoami)
cmd="time singularity exec --nv -B /home/"$username","$1"/qefm,/discover/nobackup/"$username"  "$1"/../containers/"$2" python -u -m torch.distributed.run "$1"/qefm/models/src/FMAifs/aifs-gpu-inference.py"
echo $fm: $cmd
$cmd
