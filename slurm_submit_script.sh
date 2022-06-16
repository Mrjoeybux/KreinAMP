#!/bin/bash
# Load relevant packages and modules e.g. I load a conda environment
module purge
source /gpfs01/home/psxjr4/.bashrc
source activate krein
module load gcc-uoneasy/7.3.0-2.30

# Set working_dir to the location of the AMPKrein directory
working_dir=/gpfs01/home/psxjr4/coding/AMPkrein/
cd $working_dir


command_to_execute=${@: 1}
echo "RUNNING SCRIPT"
eval $command_to_execute