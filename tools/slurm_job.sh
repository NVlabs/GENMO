user=${1:-jiefengl}
branch=${2:-main}
env_var=${3}
python_cmd=${@:4}


echo "slurm_job_id: $SLURM_JOB_ID"
echo "slurm_job_name: $SLURM_JOB_NAME"
echo "env_var: $env_var"
echo "python_cmd: $python_cmd"
echo "user: $user"
echo "branch: $branch"
echo "SUBMIT_GPUS: $SUBMIT_GPUS"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SUBMIT_SAVE_ROOT: $SUBMIT_SAVE_ROOT"

source tools/slurm_init.sh $user

echo "==========="
echo "pwd:"
pwd
echo "cmd:"
echo "$python_cmd"

if [ -n "$env_var" ]; then
    export $env_var
fi

$python_cmd
