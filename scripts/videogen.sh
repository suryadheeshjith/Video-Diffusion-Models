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

# Matteo
# with emb - 150 k : Use only for testing
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_150k_bs96_3eps name="$(date +%F)-VIDEOGEN-baseline_150k_bs96_3eps"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_200k_bs96_3eps name="$(date +%F)-VIDEOGEN-200k_bs96_3eps_2"
####

# Mine
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_spade_avg name="$(date +%F)-VIDEOGEN-3495k_spade_avg_10_5"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_concat_avg name="$(date +%F)-VIDEOGEN-349k_concat_avg_10_5"

## Autoregress

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_spade_avg_autoregress name="$(date +%F)-VIDEOGEN-3495k_spade_avg_autoregress"

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_concat_avg_autoregress name="$(date +%F)-VIDEOGEN-733k_concat_avg_autoregress"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_200k_bs96_3eps_autoregress name="$(date +%F)-VIDEOGEN-200k_bs96_3eps_autoregress"
