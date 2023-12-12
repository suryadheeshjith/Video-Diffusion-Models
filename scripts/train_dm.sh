#!/bin/bash

# Obvious / static arguments

### Baselines
# 0 eps
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_baseline_train_0epends name="$(date +%F)-1GPU_0EpENDS_restart"

# 3 eps
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=breakout_baseline_train_3epends name="$(date +%F)-1GPU_3EpENDS_restart"


### Embeddings
# SPADE Restart
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=breakout_spade_avg_restart name="$(date +%F)-1GPU_SPADE_AVG"

# CONCAT Restart
./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_4days exp=breakout_concat_avg_restart name="$(date +%F)-1GPU_CONCAT_AVG"

# SPADE Avg
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=breakout_spade_avg name="$(date +%F)-2GPU_SPADE_AVG"

# CONCAT Avg
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000 exp=breakout_concat_avg name="$(date +%F)-2GPU_CONCAT_AVG"

# SPADE Frame Concat Emb Avg
# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_4days exp=breakout_spade_frames_concat_emb_avg name="$(date +%F)-1GPU_SPADE_FRAMES_CONCAT_EMB_AVG"


# Train test
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_20mins exp=breakout_default name="$(date +%F)-encodermax_training_test1"

# 4 GPU - Doesnt work
# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=a100 exp=breakout_default name="$(date +%F)-4GPU_SPADE_MAXa100"