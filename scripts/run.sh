#!/bin/bash

# Obvious / static arguments

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=save_dataset_from_iris name="$(date +%F)-breakouttest3"
