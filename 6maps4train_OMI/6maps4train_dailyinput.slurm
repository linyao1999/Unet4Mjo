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

for memlen4train in 30
do 
    for lead4train in {1..30}
    do 
        export lead4train
        # echo $lead30d 
        export memlen4train
        # echo $memlen
        logname4train="logOMI_dailyinput_4train_mem${memlen4train}d_lead${lead4train}"
        export logname4train
        echo $logname4train
        sbatch 6maps4train_dailyinputsub.slurm 
    done
done 

# for lead in {1..3}
# do 
#     export lead
#     echo $lead 
#     srun python3 UnetOMI_dailyinput_mem30d_lead30.py > logOMI_6maps_dailyinput_mem2d_lead$lead.txt
# done 