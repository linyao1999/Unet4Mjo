#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH -t 00:02:00
#SBATCH --output=outlog/%j.out

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

vertical_mode=1
m=5
# ysta_train=1979
yend_train=2015

ysta_test=2015
yend_test=2020

export vertical_mode
export m
# export ysta_train
export ysta_test
export yend_train
export yend_test

for lat_lim in 20 40 60
do 
    for ysta_train in 1901 1959 1979 1989 1999 2009
    do
        for memlen in 1 30  # 1 30 15 5
        do 
            for lead30d in 1 5 10 13 15 20  # {10..20}
            do 
                export lead30d
                # echo $lead30d 
                export memlen
                # echo $memlen
                export ysta_train 
                export lat_lim 

                logname="logRMM_1mapyproj_${lat_lim}deg_${m}mode_dailyinput_mem${memlen}d_${ysta_train}_lead${lead30d}"
                export logname
                echo $logname
                sbatch arunsub.slurm 
            done
        done 
    done
done 

# for lead in {1..3}
# do 
#     export lead
#     echo $lead 
#     srun python3 UnetOMI_dailyinput_mem30d_lead30.py > logOMI_6maps_dailyinput_mem2d_lead$lead.txt
# done 