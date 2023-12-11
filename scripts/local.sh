#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

./.python-greene submitit_hydra.py $comp exp=save_dataset_from_iris name="$(date +%F)-breakout_autoregressive"
