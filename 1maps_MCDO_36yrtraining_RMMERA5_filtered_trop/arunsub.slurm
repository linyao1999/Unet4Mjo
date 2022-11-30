#!/bin/bash
#SBATCH -A m1517_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 12:00:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

# varname="u200"
# export varname
# lead30d=1
# export lead30d
# memlen=1
# export memlen
# logname="logRMM_${varname}trop_dailyinput_mem${memlen}d_lead${lead30d}"
# export logname

echo $varname
echo $lead30d
echo $memlen
echo $logname
srun python3 Unet4MJOfor1v.py > ./outlog/$logname.txt
