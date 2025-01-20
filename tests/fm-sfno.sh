fm="SFNO"
container="$1"/../containers/"$2"
cd "$1"/qefm/models/src/FMSfno/torch-harmonics/torch_harmonics
export PYTHONPATH="$1"/qefm/models/src/FMSfno/torch-harmonics
current_dir=$(pwd)
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
cmd="time singularity exec --nv -B "$1"/qefm  "$1"/../containers/"$2" python -u -m torch.distributed.run "$1"/qefm/models/src/FMSfno/torch-harmonics/examples/infer_sfno_checkpoint.py"
echo $fm: $cmd
$cmd
