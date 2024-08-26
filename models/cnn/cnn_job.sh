#!/bin/bash
#$ -N cnn_job         # Job name
#$ -V                 # Export all environment variables
#$ -cwd               # Run job in the current working directory
#$ -l h_rt=01:00:00   # Request 1 hour of runtime
#$ -l h_vmem=12G       # Request 4GB of memory per core
#$ -o cnn_job.out            # output file
#$ -e cnn_job.err            # error file

# Load Conda module 
module load anaconda

# Create and activate a Conda environment
conda create -n cva_net python tensorflow scikit-learn -y
source activate cva_net

# Run your Python script
python /nobackup/sc23syjl/code/cnn.py
