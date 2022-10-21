#!/bin/bash
#SBATCH -A m1517_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH -t 00:01:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv


# Note memlen and lead30d in 6hr runs represent how many time steps.
for memlen in 1  # 1 30 15 5
do 
    for lead30d in {4..120..4} # {1..30}
    do 
        export lead30d
        # echo $lead30d 
        export memlen
        # echo $memlen
        logname="logRMM_7mapstrop_6hrinput_mem${memlen}_lead${lead30d}"
        export logname
        echo $logname
        sbatch arunsub.slurm 
    done
done 
