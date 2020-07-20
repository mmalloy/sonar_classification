#!/usr/bin/env bash
#SBATCH --job-name=ml_python_new
#SBATCH --partition=wacc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:00:10
#SBATCH --output="../../output/ml_python_example-%j.txt"
#SBATCH --constraint=skylake

cd $SLURM_SUBMIT_DIR

module load anaconda/wml
bootstrap_conda

conda activate ece697

python ../python_scripts/sonar_tf.py
