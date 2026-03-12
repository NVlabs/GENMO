# Sync script for copying the repo to a remote cluster.
# Configure REMOTE_HOST and REMOTE_PATH for your environment before use.
REMOTE_HOST="${REMOTE_HOST:-your-remote-host}"
REMOTE_PATH="${REMOTE_PATH:-/path/to/remote/destination}"

rsync -azm --partial --chmod=775 \
    --exclude="/inputs" \
    --exclude="/outputs*" \
    --exclude="/glove" \
    --exclude="**/out" \
    --exclude="**/wandb" \
    --exclude="**/__pycache__" \
    --exclude="**/doc" \
    --exclude="**/docs" \
    --exclude="*.ipynb" \
    --exclude="*.safetensors" \
    --include="gem/**" \
    --include="configs/**" \
    --include="scripts/**" \
    --include="setup.py" \
    --include="third-party/**" \
    --include="third_party/**" \
    --exclude="*/*" \
    ./ \
    "${REMOTE_HOST}:${REMOTE_PATH}"
