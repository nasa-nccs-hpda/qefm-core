echo "GraphCast":
cd /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast
export PYTHONPATH=/discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/graphcast
module load singularity
cmd="time singularity exec --nv -B /discover/nobackup/projects/QEFM/qefm-core/qefm /discover/nobackup/projects/QEFM/containers/"$1" python -u -m torch.distributed.run /discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMGraphCast/google_graphcast.py"
echo $cmd
$cmd

