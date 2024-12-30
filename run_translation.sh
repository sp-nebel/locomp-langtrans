#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --ntasks=40
#SBATCH --time=00:30:00
#SBATCH --mem=10gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=usxcp@student.kit.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=translation_job
#SBATCH --output=../translation_job.out

module load devel/miniconda
conda activate llama_test_env
pip install -r requirements.txt
python3 flores_translation_pipeline.py
conda deactivate
