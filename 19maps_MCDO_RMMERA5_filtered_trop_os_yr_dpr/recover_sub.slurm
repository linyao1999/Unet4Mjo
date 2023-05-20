#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 01:10:00

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=dasrepo_g


module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

# dropout_rate_conv=0.4
# dropout_rate_mlp=0.3
# ysta_train=1901
# ysta_test=1953
# yend_train=1902
# yend_test=1954

# export dropout_rate_conv
# export dropout_rate_mlp
# export ysta_train
# export ysta_test
# export yend_train
# export yend_test

# lead30d=1
# export lead30d
# memlen=1
# export memlen
# logname="logRMM_19mapstrop_dailyinput_mem${memlen}d_lead${lead30d}_dpr${dropout_rate_conv}_${dropout_rate_mlp}_${ysta_train}_${ysta_test}_${yend_train}_${yend_test}"
# export logname

echo $lead30d
echo $memlen
echo $logname
echo $dropout_rate_conv
echo $dropout_rate_mlp
echo $ysta_train
echo $yend_train
echo $ysta_test
echo $yend_test

srun python3 ReloadUnet4MJO_loop.py > ./outlog/$logname.txt
