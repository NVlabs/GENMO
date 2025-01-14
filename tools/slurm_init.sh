user=${1:-jiefengl}

declare -A wandb_keys=( 
    ["yey"]="ffce47cffe2c549616c2b8474b1bc02c36531194" 
    ["uiqbal"]="464ee801db6f8007fcc84a8cb0d0c5cd07a4a83b"
    ["drempe"]="cd969fa7c2159d17d536f66796c8a34adfcb36eb"
    ["haotianz"]="a8222cccae5b1338d8afe0a35cc18c4a1c0ae3d3"
    ["jiefengl"]="7ecf9f6fe9e8b7263e6141fb8fc69158c2bd7fa0"
)

# wandb login ${wandb_keys[${user}]}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# rm -rf /root/.cache
# if [ ! -d "/root/.cache" ]; then
#     ln -s /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/cache /root/.cache
# fi
cache_dir="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/cache/$user"
if [ ! -d "$cache_dir" ]; then
    mkdir -p $cache_dir
fi
export TORCH_HOME=$cache_dir
export HUGGINGFACE_HUB_CACHE=$cache_dir
export XDG_CACHE_HOME=$cache_dir

results_dir="/lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/workspaces/motiondiff/motiondiff_results/$user/gvhmr"
if [ ! -d "$results_dir" ]; then
    mkdir -p $results_dir
fi
if [ ! -d "outputs" ]; then
    ln -s $results_dir outputs
fi
if [ ! -d "./inputs" ]; then
    ln -s /lustre/fsw/portfolios/nvr/projects/nvr_torontoai_humanmotionfm/datasets/GVHMR ./inputs
fi

if [[ -n "$LOCAL_RANK" && "$LOCAL_RANK" -ne 0 ]]; then
    # Place the commands you want to run here
    echo "sleep 30s since LOCAL_RANK is not 0"
    sleep 30
fi

pip install transformers==4.41.2 moviepy imageio einops dnspython numpy==1.23.5 pyvista scenepic fasteners

# pip install -e third-party/DPVO
pip install -e .
