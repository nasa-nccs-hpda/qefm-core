date
START=$(date +%s)
path=$1
container=$2
time ./tests/fm-aurora.sh $path $container
time ./tests/fm-fourcastnet.sh $path $container
time ./tests/fm-graphcast.sh $path $container
time ./tests/fm-pangu.sh $path $container
time ./tests/fm-privthi.sh $path $container
time ./tests/fm-sfno.sh $path $container
date
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
