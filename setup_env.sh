module load singularity/3.2.0
module load cuda/10.0
singularity shell -H /home/hurwit2/scratch/habitat --nv -B /usr/local/cuda --writable ~/scratch/habitat
cd habitat-api
git checkout stable
pip install -e .
cd ../habitat-sim
git checkout stable
python setup.py install --headless
