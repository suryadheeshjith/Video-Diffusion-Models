#!/bin/bash

# Obvious / static arguments

### Sampling
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_default name="$(date +%F)-breakoutsampling"

# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_2hrs exp=breakout_default name="$(date +%F)-breakoutsamplingmultigputest"
