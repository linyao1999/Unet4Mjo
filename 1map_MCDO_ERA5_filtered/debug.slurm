#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH -t 00:05:00
#SBATCH --output=outlog/%j.out

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=dasrepo_g


module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

ysta_train=1979
yend_train=1980
ysta_test=2019
yend_test=2020

lat_lim=15
leadmjo=30
nmem=1
dmem=1

mjo_ind='ROMI'

export ysta_train
export ysta_test
export yend_train
export yend_test

export lat_lim
export leadmjo
export nmem
export dmem

export mjo_ind


srun python3 Unet4MJO_loop.py > ./outlog/$logname.txt
