#!/bin/bash
#SBATCH -J bilevel                       # job name
#SBATCH -A kwang692                      # account to which job is charged, ex: GT-gburdell3
#SBATCH -N 1 --ntasks-per-node=1     	 # number of nodes and cores per node required
#SBATCH --mem-per-cpu=2G                 # memory per core
#SBATCH -t 180                            # duration of the job (ex: 10 min)

cd /storage/home/hcoda1/9/kwang692/p-kwang692-0/constrained-bilevel-optimization

module load anaconda3
conda activate bilevel

python3 main.py --solver=$SOLVER --seed=$SEED --ydim=$YDIM --n-constraints=$NCON --lr=0.01
