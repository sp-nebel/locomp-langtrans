#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --ntasks=40
#SBATCH --time=00:40:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=lora_job
#SBATCH --output=../lora_job.out

module load devel/miniconda
conda activate llama_test_env
pip install -r requirements.txt
python3 xnli_lora_training.py
conda deactivate
