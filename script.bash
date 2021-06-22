#!/bin/bash

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
#conda activate $CONDA_ENV

# jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10 --no-browser --ip=$(hostname -I) --port-retries=100
