#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 05:30:00
#SBATCH --output=outlog/%j.out

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=dasrepo_g


module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

# lead30d=1
# export lead30d
# memlen=1
# export memlen
# logname="logRMM_7mapstrop_dailyinput_mem${memlen}d_lead${lead30d}"
# export logname

echo $lead30d
echo $memlen
echo $logname
echo $mjo_ind
srun python3 Unet4MJO_loop.py > ./outlog/$logname.txt
