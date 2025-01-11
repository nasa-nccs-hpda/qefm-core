
fm="FourCastNet"
container="$1"/../containers/"$2"
cd "$1"/qefm/models/src/FMFourCastNet/FourCastNet
current_dir=$(pwd)
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

module load singularity
cmd="time singularity exec --nv -B "$1"/qefm  "$1"/../containers/"$2" python -u -m torch.distributed.run "$1"/qefm/models/src/FMFourCastNet/FourCastNet/inference/inference.py --config=afno_backbone --run_num=0 --weights "$1"/qefm/models/checkpoints/FMFourCastNet/ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt --override ./out/20240111_1605 --vis"
echo $fm: $cmd
$cmd
