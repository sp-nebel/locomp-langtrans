#!/bin/bash
#SBATCH --partition=dev_gpu_4
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:20:00
#SBATCH --mem=8gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=parrot_job
#SBATCH --output=../parrot_job.out

module load devel/miniconda
module load devel/cuda
conda activate llama_test_env
pip install -r requirements.txt
source run_scripts/run_parrot_command.sh
conda deactivate
