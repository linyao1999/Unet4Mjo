#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH -t 00:02:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

for memlen in 1  # 1 30 15 5
do 
    for lead30d in 1 5 10 13 15 20 25 30  # {10..20}
    do 
        export lead30d
        # echo $lead30d 
        export memlen
        # echo $memlen
        logname="logRMM_MCDO_19mapstrop_dailyinput_mem${memlen}d_lead${lead30d}"
        export logname
        echo $logname
        sbatch arunsub.slurm 
    done
done 

# for lead in {1..3}
# do 
#     export lead
#     echo $lead 
#     srun python3 UnetOMI_dailyinput_mem30d_lead30.py > logOMI_6maps_dailyinput_mem2d_lead$lead.txt
# done 