#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 04:10:00
#SBATCH --output=outlog/%j.out

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

# vertical_mode=1
# m=5
# mflg='odd'
# lat_lim=20
# ysta_train=1979
# yend_train=1980

# ysta_test=2015
# yend_test=2016

# export vertical_mode
# export m
# export mflg
# export lat_lim
# export ysta_train
# export ysta_test
# export yend_train
# export yend_test

# lead30d=1
# export lead30d
# memlen=1
# export memlen
# logname="logRMM_1mapyproj_${lat_lim}deg_m${m}_${mflg}_dailyinput_mem${memlen}d_lead${lead30d}"
# export logname

# srun python3 ReloadUnet4MJO_mcdp.py > ./outlog/$logname.txt
srun python3 Unet4MJO_loop.py > ./outlog/$logname.txt
