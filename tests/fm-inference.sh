fm="Inference"
current_dir=$(pwd)
# $1 = <container name>, $2 <fm name>
cmd="$current_dir/tests/fm-$2.sh $current_dir $1"
echo $fm: $cmd
$cmd
