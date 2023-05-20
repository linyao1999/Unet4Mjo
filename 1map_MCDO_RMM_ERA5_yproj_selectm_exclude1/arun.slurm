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
# m=5
mflg='ex1'
ysta_train=1979
yend_train=2015

ysta_test=2015
yend_test=2020

export vertical_mode
# export m
export mflg
export ysta_train
export ysta_test
export yend_train
export yend_test

for lat_lim in 20 
do 
    for m in 5
    do 
        for memlen in 1 30  # 1 30 15 5
        do 
            for lead30d in 1 5 10 13 15 20 25 30  # {10..20}
            do 
                export lead30d
                # echo $lead30d 
                export memlen
                # echo $memlen
                export m
                export lat_lim

                logname="logRMM_olryproj_${lat_lim}deg_m${m}_${mflg}_dailyinput_mem${memlen}d_lead${lead30d}"
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