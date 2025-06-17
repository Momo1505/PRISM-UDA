#!/bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/conv_and_attention_unet_star/GTAtoCityscapes_smothing.out
hostname
module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version

cd ~/domain_adaptation
git checkout conv_and_attention_unet_star

python run_experiments.py --config configs/mic/GTAtoCityscapes_smothing.py
echo 'END'
