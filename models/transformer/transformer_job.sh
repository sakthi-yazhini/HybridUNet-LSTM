#!/bin/bash
#$ -N transformer_job         # Job name
#$ -V                 # Export all environment variables
#$ -cwd               # Run job in the current working directory
#$ -l h_rt=48:00:00   # Request hour of runtime
#$ -l h_vmem=140G       # Request n GB of memory per core
#$ -o transformer3_job.out            # output file
#$ -e transformer3_job.err            # error file

# Load Conda module 
module load anaconda

# Create and activate a Conda environment
conda create -n cva_net python tensorflow scikit-learn -y
source activate cva_net

# Run your Python script
python /nobackup/sc23syjl/code/transformer.py
