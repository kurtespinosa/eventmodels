#!/bin/bash --login
#$ -cwd             # Job will run from the current directory
#$ -pe smp.pe 16    # Number of cores (can be 2 -- 32)


module load apps/anaconda3/5.2.0            # Python 3.6.5
module load apps/binapps/chainer/5.4.0

# Only use the requested number of CPU cores. $NSLOTS is set to the number above.
export OMP_NUM_THREADS=$NSLOTS

python ../event-model/main.py --yaml ../yaml/exm_tees_gold_cg13_test_05_02.yaml --test
