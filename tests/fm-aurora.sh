echo "Aurora":
cd /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMAurora
module load singularity
cmd="time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core/qefm /discover/nobackup/projects/QEFM/containers/"$1" python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMAurora/predictions-for-ERA5.py"
echo $cmd
$cmd
