#!/bin/bash

# Obvious / static arguments

### Training
# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=rtx8000 exp=breakout name="$(date +%F)-4GPU_SPADE_AVG2"

# Baselines
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_baseline_0epends name="$(date +%F)-1GPU_0EpENDS"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_baseline_3epends name="$(date +%F)-1GPU_3EpENDS"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=breakout_spade_avg_restart name="$(date +%F)-1GPU_SPADE_AVG"

# 1
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=breakout_spade_avg name="$(date +%F)-2GPU_SPADE_AVG"

# 2
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=breakout_concat_avg name="$(date +%F)-2GPU_CONCAT_AVG"

# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=a100 exp=breakout name="$(date +%F)-4GPU_SPADE_MAXa100"

# Train test
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_20mins exp=breakout name="$(date +%F)-encodermax_training_test1"

### Sampling
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout name="$(date +%F)-breakoutsampling"

# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_2hrs exp=breakout name="$(date +%F)-breakoutsamplingmultigputest"

### Videogen

# Baselines
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_0eps name="$(date +%F)-VIDEOGEN-baseline_87k_bs32_0eps"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_3eps name="$(date +%F)-VIDEOGEN-baseline_87k_bs32_3eps"


## Matteo Baselines
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_600k_bs32_0eps name="$(date +%F)-VIDEOGEN-baseline_600k_bs32_0eps"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_150k_bs128_3eps name="$(date +%F)-VIDEOGEN-baseline_150k_bs128_3eps"

# with emb - 150 k : Use only for testing
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_150k_bs96_3eps name="$(date +%F)-VIDEOGEN-baseline_150k_bs96_3eps"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_200k_bs96_3eps name="$(date +%F)-VIDEOGEN-baseline_200k_bs96_3eps"
####

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_spade_avg name="$(date +%F)-VIDEOGEN-72k_spade_avg"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout name="$(date +%F)-VIDEOGEN-checkpoint_300K_bs128_3epsends_10"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_noemb_0eps name="$(date +%F)-VIDEOGEN-run2_800k_bs32_noemb_0eps_10samp_5traj"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_noemb_3eps name="$(date +%F)-VIDEOGEN-300k_bs128_noemb_3eps_10samp_5traj"