#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=v100:8
#SBATCH --mem=64G

export XDG_RUNTIME_DIR="" node=$(hostname -s)
user=$(whoami)
submit_host=${SLURM_SUBMIT_HOST}
port=8888

echo $node pinned to port $port

module load cuda

cd /ibex/scratch/somc
source /home/${USER}/.bashrc
conda activate env

echo -e "
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need
to run the following two commands in a terminal.
Command to create ssh tunnel from your workstation/laptop to glogin:

ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa

Copy the link provided below by jupyter-server and replace the NODENAME with localhost before
pasting it in your browser on your workstation/laptop "

jupyter lab --no-browser --port=${port} --ip=${node} --ServerApp.token=''
