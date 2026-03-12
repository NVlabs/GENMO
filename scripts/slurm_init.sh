export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Local developer overrides — never committed (see .gitignore)
if [[ -f "$(dirname "$0")/slurm_init.local.sh" ]]; then
    source "$(dirname "$0")/slurm_init.local.sh"
fi

# Configure the variables below for your cluster environment.
# CACHE_DIR    : directory for PyTorch / HuggingFace model caches
# RESULTS_DIR  : directory where training outputs will be written
# DATASET_DIR  : path to the GEM dataset
CACHE_DIR="${CACHE_DIR:-/path/to/cache/$user}"
RESULTS_DIR="${RESULTS_DIR:-/path/to/results}"
DATASET_DIR="${DATASET_DIR:-/path/to/data}"

export TORCH_HOME=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export XDG_CACHE_HOME=$CACHE_DIR


if [[ -n "$SLURM_PROCID" && "$SLURM_LOCALID" -ne 0 ]]; then
    echo "skip installation since SLURM_LOCALID is not 0"
    # Check if the total number of SLURM nodes used is more than 4
    if [ "$SLURM_JOB_NUM_NODES" -gt 4 ]; then
        echo "sleep 60s since SLURM_JOB_NUM_NODES is more than 4"
        sleep 60
    else
        echo "sleep 60s since SLURM_JOB_NUM_NODES is less than 4"
        sleep 60
    fi
else
    echo "run installation since SLURM_PROCID is 0"

    if [ ! -d "$CACHE_DIR" ]; then
        mkdir -p $CACHE_DIR
    fi

    mkdir -p $RESULTS_DIR
    ln -s $RESULTS_DIR outputs
    ln -s $DATASET_DIR ./inputs

    pip install yacs
    apt-get update && apt-get install -y libgl1-mesa-dev libosmesa6-dev
    pip install pyrender
    pip install --upgrade PyOpenGL PyOpenGL_accelerate
    pip install -e .

fi
