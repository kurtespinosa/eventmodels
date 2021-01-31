#!/bin/bash --login
#$ -cwd             # Job will run from the current directory
#$ -pe smp.pe 4    # Number of cores (can be 2 -- 32)
#$ -l mem256 


module load apps/anaconda3/5.2.0            # Python 3.6.5
module load apps/binapps/chainer/5.4.0

# Only use the requested number of CPU cores. $NSLOTS is set to the number above.
export OMP_NUM_THREADS=$NSLOTS

python ../src/main.py --yaml ../yaml/sbnn.yaml