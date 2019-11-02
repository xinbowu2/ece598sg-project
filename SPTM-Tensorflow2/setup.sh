#!/usr/bin/env bash
conda env create -f conda_env.txt
source activate sptm
pip3 install vizdoom==1.1.4
pip3 install omgifol==0.3.0
