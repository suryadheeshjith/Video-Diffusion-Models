#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# test mae crop
./.python-greene submitit_hydra.py $comp exp=breakout name="$(date +%F)-local-encoder_mask_train_test"
