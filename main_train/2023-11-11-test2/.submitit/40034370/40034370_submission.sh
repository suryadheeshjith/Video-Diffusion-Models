#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=12
#SBATCH --error=/scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/main_train/2023-11-11-test2/.submitit/%j/%j_0_log.err
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=2023-11-11-test2
#SBATCH --mem=90GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/main_train/2023-11-11-test2/.submitit/%j/%j_0_log.out
#SBATCH --signal=USR2@120
#SBATCH --time=120
#SBATCH --wckey=submitit

# setup
cd /scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/main_train/2023-11-11-test2/.snapshot
export DATA_OVERLAY=
export RESUBMIT_COUNT=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)-ib0
export MASTER_PORT=$(for port in $(shuf -i 30000-65500 -n 20); do if [[ $(netstat -tupln 2>&1 | grep $port | wc -l) -eq 0 ]] ; then echo $port; break; fi; done;)

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/main_train/2023-11-11-test2/.submitit/%j/%j_%t_log.out --error /scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/main_train/2023-11-11-test2/.submitit/%j/%j_%t_log.err --cpu-bind=verbose \
 /scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/.resubmit.sh \
 /scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/.python-greene \
 -u -m submitit.core._submit /scratch/sd5313/CILVR/fall23/LLVM/DiffusionModels/main_train/2023-11-11-test2/.submitit/%j
