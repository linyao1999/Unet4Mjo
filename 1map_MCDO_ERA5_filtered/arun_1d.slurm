#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
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

nmem=1
dmem=1
export nmem
export dmem

for lat_lim in 20
do 
    for mjo_ind in 'RMM' 'ROMI'
    do
        for leadmjo in 1 5 10 13 15 20 25 30
        for varname in "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "sst" "olr" "tcwv" "prep"
        do 
            # export leadmjo
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