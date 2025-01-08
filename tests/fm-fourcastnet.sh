
fm="FourCastNet"
container="$1"/containers/"$2"
cd "$1"/qefm/models/src/FMFourCastNet/FourCastNet
current_dir=$(pwd)
if [[ ! -z "${PYTHONPATH}" ]]; then
    echo "PYTHONPATH: "$PYTHONPATH""
fi

# echo "FourCastNet":
# module load singularity
# cd /explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/src/FMFourCastNet/FourCastNet

module load singularity
cmd="time singularity exec --nv -B "$1"/qefm  "$1"/containers/"$2" python -u -m torch.distributed.run /explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/src/FMFourCastNet/FourCastNet/inference/inference.py --config=afno_backbone --run_num=0 --weights /explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/checkpoints/FMFourCastNet/ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt --override ./out/20240108_1649 --vis"
# cmd="time singularity exec --nv -B /explore/nobackup/projects/ilab/projects/qefm-core /explore/nobackup/projects/ilab/projects/qefm-core/containers/"$1" python -u -m torch.distributed.run /explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/src/FMFourCastNet/FourCastNet/inference/inference.py --config=afno_backbone --run_num=0 --weights /explore/nobackup/projects/ilab/projects/qefm-core/qefm/models/checkpoints/FMFourCastNet/ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt --override ./out/20240103_1555 --vis"
echo $fm: $cmd
$cmd
