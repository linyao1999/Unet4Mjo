#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=1
#SBATCH -t 04:10:00
#SBATCH --output=outlog/submit-%j.out

# source $LMOD_PKG/init/csh
# # module load esslurm
# # module load cgpu
# # module load python

module load pytorch/1.11.0
# module load cudatoolkit
# conda activate eofenv

# dropout_rate_conv=0.1
# dropout_rate_mlp=0.2

# ysta_train=1979
# yend_train=1980

# ysta_test=2015
# yend_test=2016

# mon1=12
# monlen=4
# numepo=2
# Nsamp=10

# export dropout_rate_conv
# export dropout_rate_mlp
# export ysta_train
# export ysta_test
# export yend_train
# export yend_test
# export mon1
# export monlen
# export numepo
# export Nsamp

# lead30d=1
# export lead30d
# memlen=1
# export memlen
# logname="19mapstrop_dailyinput_mem${memlen}d_lead${lead30d}_dpr${dropout_rate_conv}_${dropout_rate_mlp}_${ysta_train}_${ysta_test}_${yend_train}_${yend_test}_${mon1}"
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
echo $mon1
echo $monlen

srun python3 Unet4MJO_loop.py > ./outlog/$logname.txt
