#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# test mae crop
# ./.python-greene submitit_hydra.py $comp exp=breakout_videogen_concat_avg_autoregress name="$(date +%F)-testVIDEOGEN-1_4M_concat_avg_autoregress2"
# ./.python-greene submitit_hydra.py $comp exp=save_dataset_from_iris name="$(date +%F)-breakout_autoregressive"

./.python-greene submitit_hydra.py $comp exp=breakout_iris_decoded name="$(date +%F)-test_IRIS_DECODED"