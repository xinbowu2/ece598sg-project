source activate habitat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xinbowu2/scratch/ece598sg-project/cuda/lib64
cd ece598sg-project/SMT
python training_policy_with_env_wrapper_al.py --cfg experiments/smt_gibson_debug.yaml 
