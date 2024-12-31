# PanguWeather:
echo "PanguWeather:"
cd /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMPangu/Pangu-Weather-pytorch
module load singularity
cmd="time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core/qefm /discover/nobackup/projects/QEFM/containers/"$1" python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMPangu/Pangu-Weather-pytorch/inference_gpu.py"
echo $cmd
$cmd
#time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core/qefm /discover/nobackup/projects/QEFM/containers/qefm-20241226.sif python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMPangu/Pangu-Weather-pytorch/inference_gpu.py

