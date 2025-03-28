#!/bin/bash

source $MY_ENVS/ML-Docking/bin/activate
export DRIVER_PATH="/home/ht1410/MLDocking_Dragon/Dragon/"
export DATA_PATH="/flare/datascience/dragon/tiny/"
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,aurora-adminvm-01,localhost,*.cm.aurora.alcf.anl.gov,aurora-*,*.aurora.alcf.anl.gov,*.alcf.anl.gov"

if [ ! -d "./ensemble_launcher" ]; then
    git clone --branch dev https://github.com/argonne-lcf/ensemble_launcher.git
fi

python3 launch_inference_el.py

# start=$SECONDS
# mpirun -np 1 -ppn 1 --depth 1 gpu_tile_compact.sh python3 run_inference.py --nps 12 --pid 0
# elapsed=$(( SECONDS - start ))
# echo "Execution time with direct command: $elapsed seconds"

# start=$SECONDS
# python3 test_subprocess.py
# elapsed=$(( SECONDS - start ))
# echo "Execution time with subprocess: $elapsed seconds"

