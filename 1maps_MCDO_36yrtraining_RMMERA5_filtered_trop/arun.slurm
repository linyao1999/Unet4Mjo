#!/bin/bash
#SBATCH -A m1517_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH -t 00:02:00

# this script is to use 1 variable to predict MJO
# 1-day memory

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

# variable order:
# 0: u200
# 1: u850
# 2: olr
# 3: tcwv
# 4: v200
# 5: T200
# 6: sst

# varname="u200"
# export varname

for memlen in 1  # 1 30 15 5
do 
    for lead30d in 1 5 10 13 15 20 25 30  # {10..20}
    do 
        for varname in "tcwv" "v200" "T200" "sst"  # "u200" "u850" "olr" "tcwv" "v200" "T200" "sst"
        do 
            export varname
            export lead30d
            # echo $lead30d 
            export memlen
            # echo $memlen
            logname="logRMM_MCDO_${varname}trop_dailyinput_mem${memlen}d_lead${lead30d}"
            export logname
            echo $logname
            sbatch arunsub.slurm 
        done
    done
done 

# for lead in {1..3}
# do 
#     export lead
#     echo $lead 
#     srun python3 UnetOMI_dailyinput_mem30d_lead30.py > logOMI_6maps_dailyinput_mem2d_lead$lead.txt
# done 