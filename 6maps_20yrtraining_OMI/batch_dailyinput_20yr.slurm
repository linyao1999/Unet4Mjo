#!/bin/bash
#SBATCH -A m1517_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 00:20:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

for memlen20yr in 30 15 5 1
do 
    for lead20yr in {1..30}
    do 
        export lead20yr
        # echo $lead30d 
        export memlen20yr
        # echo $memlen
        logname20yr="logOMI_6maps_dailyinput_20yr_mem${memlen20yr}d_lead${lead20yr}"
        export logname20yr
        echo $logname20yr
        sbatch batch_dailyinput_20yrsub.slurm 
    done
done 

# for lead in {1..3}
# do 
#     export lead
#     echo $lead 
#     srun python3 UnetOMI_dailyinput_mem30d_lead30.py > logOMI_6maps_dailyinput_mem2d_lead$lead.txt
# done 