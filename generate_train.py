import os

# Directories
config_dir = "configs/mic"
output_dir = "slurm_scripts"
launcher_script_path = "launch_train.sh"

##SBATCH -p grantgpu -A g2024a219g
##SBATCH -p publicgpu -A miv

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Base SLURM script template
slurm_template = """#!/bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/{job_name}.out
hostname
module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version

cd ~/domain_adaptation
python run_experiments.py --config {config_path}
echo 'END'
"""

# Initialize the launcher script
with open(launcher_script_path, "w") as launcher_file:
    launcher_file.write("#!/bin/bash\n\n")  # Shebang and newline

# Iterate over all configuration files in the config directory
for config_file in os.listdir(config_dir):
    if config_file.endswith(".py"):  # Ensure it's a Python config file
        config_path = os.path.join(config_dir, config_file)
        job_name = os.path.splitext(config_file)[0]  # Remove file extension for job name
        
        # Create the SLURM script content
        slurm_script = slurm_template.format(job_name=job_name, config_path=config_path)
        
        # Define the output file path
        slurm_output_path = os.path.join(output_dir, f"{job_name}.slurm")
        
        # Write the SLURM script to the file
        with open(slurm_output_path, "w") as f:
            f.write(slurm_script)
        
        # Append a line to the launcher script to submit the SLURM job
        with open(launcher_script_path, "a") as launcher_file:
            launcher_file.write(f"sbatch {slurm_output_path}\n")

# Make the launcher script executable
os.chmod(launcher_script_path, 0o755)

print(f"SLURM scripts generated in '{output_dir}' directory.")
print(f"Launcher script created at '{launcher_script_path}'.")