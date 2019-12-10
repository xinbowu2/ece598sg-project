source activate habitat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xinbowu2/scratch/ece598sg-project/cuda/lib64
python debug_smt.py --cfg experiments/smt_gibson_debug.yaml 
