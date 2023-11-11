#!/bin/bash

# Obvious / static arguments

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=main_train name="$(date +%F)-test2"
