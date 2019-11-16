#!/bin/sh
source activate habitat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dpshah2/cuda/lib64
cd src/
python -m train.train_action_predictor