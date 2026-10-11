#!/bin/bash
#SBATCH --job-name=ms5_sorting
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fastdata/home/libin/data/sorting_%j.out
#SBATCH --error=/fastdata/home/libin/data/sorting_%j.err

source /fastdata/apps/anaconda3/etc/profile.d/conda.sh
conda activate myenv

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

cd /fastdata/home/libin/data
python -u sorting2.py