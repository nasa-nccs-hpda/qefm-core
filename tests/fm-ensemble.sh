date
START=$(date +%s)
time /discover/nobackup/projects/QEFM/qefm-core/tests/fm-aurora.sh $1
time /discover/nobackup/projects/QEFM/qefm-core/tests/fm-fourcastnet.sh $1
time /discover/nobackup/projects/QEFM/qefm-core/tests/fm-graphcast.sh $1
time /discover/nobackup/projects/QEFM/qefm-core/tests/fm-pangu.sh $1
time /discover/nobackup/projects/QEFM/qefm-core/tests/fm-privthi.sh $1
date
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
