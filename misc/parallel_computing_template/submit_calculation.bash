#!/bin/bash
#SBATCH --job-name=template_test
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:10:00
#SBATCH --array=1-100
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

conda activate synthetic_dim

# Ensure the output directory exists (in the parent directory)
mkdir -p output

# Get the list of Python files in the 'input' folder
INPUT_FILES=($(ls input/*.py))

FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}

# Run the script and save to output/output_1.txt, output_2.txt, etc.
python $FILE > output/output_${SLURM_ARRAY_TASK_ID}.txt 2>&1
