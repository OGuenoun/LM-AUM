#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu
#SBATCH --ntasks-per-node=1



module load python/3.11 gcc arrow
source LM_env/bin/activate
cd ~/LM-AUM
python LM_train_cluster.py Cross-entropy