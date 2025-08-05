#!/bin/bash
#SBATCH --job-name=lm_train_micro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1                       
#SBATCH --output=logs/%x-%j.out    
#SBATCH --error=logs/%x-%j.err     


module load python/3.11 gcc arrow
source LM_env/bin/activate
cd ~/LM-AUM
python LM_train_cluster.py AUM_micro