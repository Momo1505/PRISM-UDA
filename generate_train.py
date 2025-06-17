import os

# Configs and output settings
config_dir = "configs/mic"
output_base_dir = "slurm_scripts"
launcher_script_path = "launch_train.sh"

branches = {
    "normal_training": "conv_and_attention_unet",  
    "modified_training": "conv_and_attention_unet_star"
}

# Base SLURM script template
slurm_template = """#!/bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/{branch}/{job_name}.out
hostname
module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version

cd ~/domain_adaptation
git checkout {branch}

python run_experiments.py --config {config_path}
echo 'END'
"""

# Initialize the launcher script
with open(launcher_script_path, "w") as launcher_file:
    launcher_file.write("#!/bin/bash\n\n")  # Shebang

# Generate scripts for each branch
for key, branch_name in branches.items():
    branch_output_dir = os.path.join(output_base_dir, key)
    os.makedirs(branch_output_dir, exist_ok=True)

    for config_file in os.listdir(config_dir):
        if config_file.endswith(".py"):
            config_path = os.path.join(config_dir, config_file)
            job_name = os.path.splitext(config_file)[0]

            # Output path for the SLURM script
            slurm_output_path = os.path.join(branch_output_dir, f"{job_name}.slurm")

            # Fill in the SLURM template
            slurm_script = slurm_template.format(
                job_name=job_name,
                config_path=config_path,
                branch=branch_name
            )

            # Write to SLURM file
            with open(slurm_output_path, "w") as f:
                f.write(slurm_script)

            # Append submission line to the launcher script
            with open(launcher_script_path, "a") as launcher_file:
                launcher_file.write(f"sbatch {slurm_output_path}\n")

# Make launcher executable
os.chmod(launcher_script_path, 0o755)

print(f"SLURM scripts generated for branches: {', '.join(branches.values())}")
print(f"Launcher script created at '{launcher_script_path}'.")
