#!/bin/bash
#SBATCH -A m1517_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 06:10:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv


# for lead in {1..3}
# do 
#     export lead
#     echo $lead 
#     srun python3 UnetOMI_dailyinput_mem30d_lead30.py > logOMI_6maps_dailyinput_mem2d_lead$lead30d.txt
# done 

# lead10yr=30
# export lead10yr
echo $lead10yr
# memlen10yr=30
# export memlen10yr
echo $memlen10yr
# logname10yr="logOMI_6maps_dailyinput_mem${memlen10yr}d_lead${lead10yr}"
echo $logname10yr
srun python3 UnetOMI_dailyinput_6maps10yrtraining.py > ./outlog/$logname10yr.txt
