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

for memlen10yr in 30 15 5 1
do 
    for lead10yr in {1..30}
    do 
        export lead10yr
        # echo $lead30d 
        export memlen10yr
        # echo $memlen
        logname10yr="logOMI_6maps_dailyinput_10yr_mem${memlen10yr}d_lead${lead10yr}"
        export logname10yr
        echo $logname10yr
        sbatch batch_dailyinput_10yrsub.slurm 
    done
done 

# for lead in {1..3}
# do 
#     export lead
#     echo $lead 
#     srun python3 UnetOMI_dailyinput_mem30d_lead30.py > logOMI_6maps_dailyinput_mem2d_lead$lead.txt
# done 