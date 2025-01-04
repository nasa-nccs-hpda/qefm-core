echo "FourCastNet":
module load singularity
cd /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMFourCastNet/FourCastNet
echo 'cd /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMFourCastNet/FourCastNet'

#cd /discover/nobackup/projects/QEFM/qefm-core
#echo 'cd /discover/nobackup/projects/QEFM/qefm-core'
cmd="time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core /discover/nobackup/projects/QEFM/containers/"$1" python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMFourCastNet/FourCastNet/inference/inference.py --config=afno_backbone --run_num=0 --weights /discover/nobackup/projects/QEFM/qefm-core/qefm/models/checkpoints/FMFourCastNet/ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt --override ./out/20240103_1555 --vis"
echo $cmd
$cmd
