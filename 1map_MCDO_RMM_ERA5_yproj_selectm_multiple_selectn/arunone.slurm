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
m=1
mflg='off'
wnxflg='one'
ysta_train=1979
yend_train=2015

ysta_test=2015
yend_test=2020

export vertical_mode
export m
export mflg
export wnxflg
export ysta_train
export ysta_test
export yend_train
export yend_test

for lat_lim in 20 
do 
    for wnx in {21..50}  # {11..20} # 2 3 4 5 6 7 8 9 10
    do 
        for memlen in 1 # 30  # 1 30 15 5
        do 
            for lead30d in 10 # 1 5 10 13 15 20 25 30  # {10..20}
            do 
                export lead30d
                # echo $lead30d 
                export memlen
                # echo $memlen
                export wnx
                export lat_lim

                logname="logRMM_1mapyproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}"
                export logname
                echo $logname
                sbatch arunsub.slurm 
            done
        done 
    done 
done
