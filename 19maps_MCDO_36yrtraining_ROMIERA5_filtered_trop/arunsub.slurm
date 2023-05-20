#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -t 03:10:00
#SBATCH --output=outlog/%j.out

source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0

# memlen=1  # 1 30 15 5
# lead30d=10
# export lead30d
# export memlen
# logname="logROMI_MCDO_19mapstrop_dailyinput_mem${memlen}d_lead${lead30d}"
# export logname

srun python3 ReloadUnet4MJO_mcdp.py > ./outlog/$logname.txt
