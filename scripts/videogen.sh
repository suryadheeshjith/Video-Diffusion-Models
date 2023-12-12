#!/bin/bash

# Obvious / static arguments

### Videogen

#######
# Baselines
#######
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_baseline_videogen_0eps name="$(date +%F)-VIDEOGEN-baseline_87k_bs32_0eps"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_baseline_videogen_3eps name="$(date +%F)-VIDEOGEN-baseline_100k_bs32_3eps"

## Matteo Baselines
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_600k_bs32_0eps name="$(date +%F)-VIDEOGEN-baseline_600k_bs32_0eps"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_baseline_150k_bs128_3eps name="$(date +%F)-VIDEOGEN-baseline_150k_bs128_3eps"

######
# With Token embeddings
#######
# with emb - 150 k : Use only for testing
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_150k_bs96_3eps name="$(date +%F)-VIDEOGEN-baseline_150k_bs96_3eps"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_200k_bs96_3eps name="$(date +%F)-VIDEOGEN-200k_bs96_3eps_2"
####

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_spade_avg name="$(date +%F)-VIDEOGEN-168k_spade_avg"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_concat_avg name="$(date +%F)-VIDEOGEN-168k_concat_avg"

## Autoregress

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_spade_avg_autoregress name="$(date +%F)-VIDEOGEN-168k_spade_avg_autoregress"

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_concat_avg_autoregress name="$(date +%F)-VIDEOGEN-300k_concat_avg_autoregress_100pred"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_200k_bs96_3eps_autoregress name="$(date +%F)-VIDEOGEN-200k_bs96_3eps_autoregress"
