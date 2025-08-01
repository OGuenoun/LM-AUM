#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu
#SBATCH --ntasks-per-node=1



module load python/3.11
source LM_env/bin/activate
python ~/LM_AUM/LM_train_cluster.py