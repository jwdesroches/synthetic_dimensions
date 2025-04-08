#!/bin/bash
#SBATCH --job-name=param_sweep
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --array=1-10000
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

module load python

INPUT_FILE="inputs/input_${SLURM_ARRAY_TASK_ID}.json"
OUTPUT_FILE="outputs/output_${SLURM_ARRAY_TASK_ID}.txt"

python run_job.py "$INPUT_FILE" "$OUTPUT_FILE"
