#!/usr/bin/env bash
set -euo pipefail

# Start virtual display for headless rendering (no real X11)
if [[ -z "${DISPLAY:-}" ]]; then
  Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
  export DISPLAY=:99
  sleep 1
fi

HOST_UID="${HOST_UID:-}"
HOST_GID="${HOST_GID:-}"
HOST_USER="${HOST_USER:-user}"

if [[ -z "${HOST_UID}" || -z "${HOST_GID}" ]]; then
  if [[ -d /workspace ]]; then
    HOST_UID="$(stat -c %u /workspace)"
    HOST_GID="$(stat -c %g /workspace)"
  else
    HOST_UID="${HOST_UID:-1000}"
    HOST_GID="${HOST_GID:-1000}"
  fi
fi

if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
  groupadd -g "${HOST_GID}" "${HOST_USER}"
fi

if ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}"
fi

exec gosu "${HOST_UID}:${HOST_GID}" "$@"
