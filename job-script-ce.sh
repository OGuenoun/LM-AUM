#!/bin/bash
#SBATCH --job-name=lm_train_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=2          
#SBATCH --output=logs/%x-%j.out    
#SBATCH --error=logs/%x-%j.err     


module load python/3.11 gcc arrow cuda cudnn
source LM_env/bin/activate
cd ~/LM-AUM
python LM_train_cluster.py Cross-entropy