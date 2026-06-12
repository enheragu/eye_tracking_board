#!/usr/bin/env bash
## Thin wrapper around run_all.py, which handles participant discovery, parallel
## execution and per-participant logs. Any argument is forwarded, e.g.:
##    ./scripts/process_video.sh -p 002 024 -j 2 -t gaze

## Path of current file (resolving symlinks); repo root is its parent folder
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
SOURCE=$(readlink "$SOURCE")
[[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
REPO_ROOT=$( dirname "$SCRIPT_PATH" )

cd "$REPO_ROOT" && python3 src/run_all.py "$@"
