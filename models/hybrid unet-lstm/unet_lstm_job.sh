#!/bin/bash
#$ -N unet_lstm_job         # Job name
#$ -V                 # Export all environment variables
#$ -cwd               # Run job in the current working directory
#$ -l h_rt=48:00:00   # hours of runtime
#$ -l h_vmem=150G       # Request n GB of memory per core
#$ -o unet_lstm_job.out            # output file
#$ -e unet_lstm_job.err            # error file

# Load Conda module 
module load anaconda

# Create and activate a Conda environment
conda create -n myenv python tensorflow scikit-learn -y
source activate myenv

# Run your Python script
python /nobackup/sc23syjl/code/unet_lstm.py
