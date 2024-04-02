#!/bin/bash
#
#SBATCH -p gpu                    # partition (queue)
#SBATCH --qos=valhala
#SBATCH -N 1                      # number of nodes
#SBATCH -n 10                     # number of cores
#SBATCH --mem=10G                # memory pool for all cores
#SBATCH -t 1-00:00                # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out        # STDOUT
#SBATCH -e slurm.%N.%j.err        # STDERR

source ../rnn_generator_env/bin/activate

python optimize.py --optimization_steps 5 --models_per_step 10