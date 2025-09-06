#!/bin/bash

wkdir="/discover/nobackup/jli30/QEFM/qefm-core/data"

cd "$wkdir"
echo $PWD

# Start and end dates
start="2022-01-01"
end="2022-12-31"

current="$start"

source /usr/share/modules/init/bash
module load anaconda
conda activate gs_download 

while [ "$(date -d "$current" +%Y%m%d)" -le "$(date -d "$end" +%Y%m%d)" ]; do
    yyyy=$(date -d "$current" +%Y)
    mm=$(date -d "$current" +%m)
    dd=$(date -d "$current" +%d)

    echo "$yyyy $mm $dd"
    cmd="python graph_input_6hr.py -y "$yyyy" -m "$mm" -d "$dd" -n 6"
 
    $cmd

    # Advance by 5 days
    current=$(date -d "$current + 5 days" +%Y-%m-%d)
done
#for DD in {01..31}; do
#	cmd="python gencast_input_6hr.py -y "$YYYY" -m "$MM" -d "$DD" -n 4"
#	echo $cmd
#
#	$cmd
#done
