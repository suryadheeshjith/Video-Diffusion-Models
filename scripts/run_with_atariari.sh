#!/bin/bash

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=run_with_atariari name="$(date +%F)-breakouttest"

