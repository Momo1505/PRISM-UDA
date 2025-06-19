#!/bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o ablations/CVCtokvasir_mic_hrda_2000.out
hostname
module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version
echo 'START'
cd ~/domain_adaptation
python run_experiments.py --config configs/mic/CVCtokvasir_mic_hrda_2000.py
echo 'END'
