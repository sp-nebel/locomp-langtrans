#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --ntasks=40
#SBATCH --time=00:02:00
#SBATCH --mem=12gb
#SBATCH --gres=gpu:1
#SBATCH --output=../dim_job.out

module load devel/miniconda
conda activate llama_test_env
pip install -r requirements.txt
python3 print_model_dims.py
conda deactivate
