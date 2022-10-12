#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=v100:8
#SBATCH --mem=64G

module load cuda

source /home/${USER}/.bashrc
conda activate env

python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 10000 --val-samples 5000 --units 512 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 10000 --val-samples 5000 --units 1024 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 50000 --val-samples 5000 --units 512 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 50000 --val-samples 5000 --units 1024 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 100000 --val-samples 5000 --units 512 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 100000 --val-samples 5000 --units 1024 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 300000 --val-samples 5000 --units 512 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/rhnn.py --train-samples 300000 --val-samples 5000 --units 1024 --embed-units 512 --epochs 50 --batch-size 1024 --target-acc .9
