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
# 3
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_200k_bs96_3eps name="$(date +%F)-VIDEOGEN-500k_bs96_3eps_10_5"
####

# Mine
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_spade_avg name="$(date +%F)-VIDEOGEN-3495k_spade_avg_10_5"
# 1
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_spade_frames_concat_emb_avg name="$(date +%F)-VIDEOGEN-1M_spade_frame_concat_emb_10_5"
# 2
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_videogen_concat_avg name="$(date +%F)-VIDEOGEN-1_4M_concat_avg_10_5"

## Autoregress

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_spade_avg_autoregress name="$(date +%F)-VIDEOGEN-3495k_spade_avg_autoregress"

# 5
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_spade_frames_concat_emb_avg_autoregress name="$(date +%F)-VIDEOGEN-1M_spade_frame_concat_emb_autoregress2"

# 4
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_concat_avg_autoregress name="$(date +%F)-VIDEOGEN-1_4M_concat_avg_autoregress2_100"

# 6
./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=breakout_videogen_200k_bs96_3eps_autoregress name="$(date +%F)-VIDEOGEN-500k_bs96_3eps_autoregress2"
