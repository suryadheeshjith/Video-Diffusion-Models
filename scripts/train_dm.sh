#!/bin/bash

# Obvious / static arguments

### Training
# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=rtx8000 exp=breakout_default name="$(date +%F)-4GPU_SPADE_AVG2"

# Baselines
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_baseline_0epends name="$(date +%F)-1GPU_0EpENDS_restart"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_baseline_3epends name="$(date +%F)-1GPU_3EpENDS_restart"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=breakout_spade_avg_restart name="$(date +%F)-1GPU_SPADE_AVG"

# 1
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=breakout_spade_avg name="$(date +%F)-2GPU_SPADE_AVG"

# 2
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=breakout_concat_avg name="$(date +%F)-2GPU_CONCAT_AVG"

# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=a100 exp=breakout_default name="$(date +%F)-4GPU_SPADE_MAXa100"

# Train test
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_20mins exp=breakout_default name="$(date +%F)-encodermax_training_test1"
