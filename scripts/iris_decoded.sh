#!/bin/bash

# Obvious / static arguments
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_iris_decoded name="$(date +%F)-IRIS_DECODED"

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_iris_decoded name="$(date +%F)-IRIS_DECODED_300"
