#!/bin/bash
#SBATCH -A m1517_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 07:00:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv


# lead30d=30
# export lead30d
echo $lead30d
# memlen=30
# export memlen
echo $memlen
# logname="logOMI_6maps_dailyinput_mem${memlen}d_lead${lead30d}"
echo $logname
srun python3 Unet4MJO.py > ./outlog/$logname.txt

# lead30d=4
# export lead30d
# echo $lead30d
# memlen=1
# export memlen
# echo $memlen
# logname="logRMM_7mapstrop_6hrinput_mem${memlen}_lead${lead30d}"
# echo $logname
# srun python3 Unet4MJO.py > ./outlog/$logname.txt