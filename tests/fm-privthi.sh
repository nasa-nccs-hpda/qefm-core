echo "Privthi/WxC":
cd /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC/examples
export PYTHONPATH=/discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC
module load singularity
cmd="time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core/qefm  /discover/nobackup/projects/QEFM/containers/"$1" python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC/examples/PrithviWxC_inference.py"
echo $cmd
$cmd
#time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core/qefm  /discover/nobackup/projects/QEFM/containers/qefm-20241226.sif python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMPrithvi-WxC/Prithvi-WxC/examples/PrithviWxC_inference.py

