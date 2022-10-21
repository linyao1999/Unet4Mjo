#!/bin/bash
#SBATCH -A m1517_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 05:10:00

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

# lead30d=30
# export lead30d
echo $lead30d
# memlen=30
# export memlen
echo $memlen
# logname="logOMI_6maps_dailyinput_mem${memlen}d_lead${lead30d}"
echo $logname
srun python3 Unet4MJO.py > ./outlog/$logname.txt
