#!/bin/bash
#
#SBATCH --partition=gpu                 # partition
#SBATCH --qos=valhala
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of cores
#SBATCH --mem=24G                       # memory per node
#SBATCH --time=5-00:00                  # time (D-HH:MM)
#SBATCH --output=slurm.%N.%j.out        # STDOUT
#SBATCH --error=slurm.%N.%j.err         # STDERR

# won't work without this
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

source ~/.bashrc
conda activate rnn_env

experiment=logs/seq_len/

srun python trainer.py --num_nodes 1 --devices 0 --train_size 1.0 --monitor_checkpoint balanced_acc/val --mode_checkpoint max --epochs 200 --path $experiment
