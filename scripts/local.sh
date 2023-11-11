#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# test mae pretrain
./.python-greene submitit_hydra.py $comp exp=main_train wandb=False name="$(date +%F)-test1"
