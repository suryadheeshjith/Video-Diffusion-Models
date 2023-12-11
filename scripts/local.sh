#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# test mae crop
./.python-greene submitit_hydra.py $comp exp=breakout_videogen_baseline_150k_bs96_3eps name="$(date +%F)-VIDEOGEN-baseline_150k_bs96_3eps"