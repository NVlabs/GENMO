user=${1:-jiefengl}

declare -A wandb_keys=(
    ["yey"]="ffce47cffe2c549616c2b8474b1bc02c36531194"
    ["uiqbal"]="464ee801db6f8007fcc84a8cb0d0c5cd07a4a83b"
    ["drempe"]="cd969fa7c2159d17d536f66796c8a34adfcb36eb"
    ["haotianz"]="a8222cccae5b1338d8afe0a35cc18c4a1c0ae3d3"
    ["jiefengl"]="7ecf9f6fe9e8b7263e6141fb8fc69158c2bd7fa0"
    ["jinkunc"]="c9a98990e2992e4a76dfad280ef8fd8bc8446caa"
)

# wandb login ${wandb_keys[${user}]}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# rm -rf /root/.cache
# if [ ! -d "/root/.cache" ]; then
#     ln -s /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/cache /root/.cache
# fi
cache_dir="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/cache/$user"
export TORCH_HOME=$cache_dir
export HUGGINGFACE_HUB_CACHE=$cache_dir
export XDG_CACHE_HOME=$cache_dir


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

    if [ ! -d "$cache_dir" ]; then
        mkdir -p $cache_dir
    fi

    results_dir="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_results/$user/gvhmr"
    mkdir -p $results_dir
    ln -s $results_dir outputs
    ln -s /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/GVHMR ./inputs

    pip install transformers==4.41.2 moviepy imageio einops dnspython numpy==1.23.5 scenepic dill librosa
    # pip install -e third-party/DPVO
    pip install git+https://github.com/google/aistplusplus_api.git
    pip install -e .
fi
