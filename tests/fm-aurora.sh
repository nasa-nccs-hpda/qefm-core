fm="Aurora"
container="$1"/../containers/"$2"
cd "$1"/qefm/models/src/FMAurora
current_dir=$(pwd)
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
cmd="time singularity exec --nv -B "$1"/qefm  "$1"/../containers/"$2" python -u -m torch.distributed.run "$1"/qefm/models/src/FMAurora/predictions-for-ERA5.py"
echo $fm: $cmd
$cmd
