#!/bin/bash
#
#SBATCH --partition=gpu                 # partition (queue)
#SBATCH --qos=valhala
#SBATCH --nodes=2                       # number of nodes
#SBATCH --ntasks-per-node=2             # number of cores
#SBATCH --mem=24G                       # memory per node
#SBATCH --time=1-00:00                  # time (D-HH:MM)
#SBATCH --output=slurm.%N.%j.out        # STDOUT
#SBATCH --error=slurm.%N.%j.err         # STDERR

# won't work without this
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

source ~/.bashrc

conda activate rnn_env

optimization_steps=50

for i in $(seq $optimization_steps)
do
    echo
    echo "-----------------------------"
    echo "Optimization step: $i / $optimization_steps"
    echo
    
    echo "Making gridsearch step."
    python gridsearch.py
    
    echo "Running trainer."
    srun python trainer.py --num_nodes 2 --devices 2 --train_size 1.0 --epochs 4000
    
    echo "Updating parameter intervals."
    python update.py --min_good_samples 4 --max_good_loss 1e-4 --check_every_n_steps 3 --current_step $i
done

echo
echo "-----------------------------"
echo "Optimization finished."