#!/bin/bash
path=$1
numitr=$2
cd $path
jid=$(sbatch --parsable jobfileNVME)

for((i=1; i<=numitr; i++)); do
    jid=$(sbatch --parsable --dependency=afterany:$jid jobfileNVME)
done
