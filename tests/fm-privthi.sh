fm="Privthi/WxC"
container="$1"/containers/"$2"
cd "$1"/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC/examples
current_dir=$(pwd)
export PYTHONPATH="$1"/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
cmd="time singularity exec --nv -B "$1"/qefm  "$1"/containers/"$2" python -u -m torch.distributed.run "$1"/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC/examples/PrithviWxC_inference.py"
echo $fm: $cmd
$cmd
