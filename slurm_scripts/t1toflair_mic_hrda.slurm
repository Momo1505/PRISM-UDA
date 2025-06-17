#!/bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/t1toflair_mic_hrda.out
hostname
module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version

cd ~/domain_adaptation
python run_experiments.py --config configs/mic/t1toflair_mic_hrda.py
echo 'END'
