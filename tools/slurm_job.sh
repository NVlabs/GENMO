user=${1:-jiefengl}
branch=${2:-main}
python_cmd=${@:3}


echo "slurm_job_id: $SLURM_JOB_ID"
echo "slurm_job_name: $SLURM_JOB_NAME"
echo "python_cmd: $python_cmd"
echo "user: $user"
echo "branch: $branch"
echo "SUBMIT_GPUS: $SUBMIT_GPUS"
echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SUBMIT_SAVE_ROOT: $SUBMIT_SAVE_ROOT"

source tools/slurm_init.sh $user

echo "==========="
echo "pwd:"
pwd
echo "cmd:"
echo "python $python_cmd"
python $python_cmd

