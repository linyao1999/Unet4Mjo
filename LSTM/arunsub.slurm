#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 00:06:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

# salloc --nodes 1 --qos interactive --time 00:03:00 --constraint gpu --gpus 1 --account=dasrepo_g arunsub.slurm
# 

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

# lead30d=30
# export lead30d
# memlen=1
# export memlen
# logname="logRMM_19mapstrop_dailyinput_mem${memlen}d_lead${lead30d}"
# export logname

mkdir -p ./outlog;

echo $lead30d
echo $memlen
echo $logname
srun python3 LSTM_NN_RMM.py > ./outlog/$logname.txt
