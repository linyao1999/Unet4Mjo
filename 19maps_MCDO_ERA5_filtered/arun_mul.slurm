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

ysta_train=1979
yend_train=2015
ysta_test=2015
yend_test=2020
export ysta_train
export ysta_test
export yend_train
export yend_test

nmem=30
dmem=10
export nmem
export dmem

for lat_lim in 20
do 
    for mjo_ind in 'RMM' 'ROMI'
    do
        for leadmjo in 1 5 10 15 20 25 30  # {10..20}
        do
            export leadmjo
            export mjo_ind
            export lat_lim
            logname="log${mjo_ind}_MCDO_19maps${lat_lim}deg_dailyinput_mem${nmem}d_dmem_${dmem}_lead${leadmjo}"
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