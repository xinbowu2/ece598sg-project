#!/bin/sh
source activate habitat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hurwit2/cuda/lib64
cd ece598sg-project/SPTM-Tensorflow2/src
python -m train.train_edge_predictor
