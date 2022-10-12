#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=v100:8
#SBATCH --mem=16G

module load cuda

source /home/${USER}/.bashrc
conda activate env

python /ibex/scratch/somc/AI-Blog/code/conv_transformer.py --train-samples 100000 --val-samples 5000 --num-layers 6 --d-model 512 --dff 2048 --num-heads 8 --dropout-rate .1 --epochs 50 --batch-size 512 --target-acc .9
python /ibex/scratch/somc/AI-Blog/code/conv_transformer.py --train-samples 300000 --val-samples 5000 --num-layers 6 --d-model 512 --dff 2048 --num-heads 8 --dropout-rate .1 --epochs 50 --batch-size 512 --target-acc .9
