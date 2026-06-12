#!/usr/bin/env bash
## Projects eye data over the original videos for the given participants.

## Path of current file (resolving symlinks); repo root is its parent folder
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
SOURCE=$(readlink "$SOURCE")
[[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
REPO_ROOT=$( dirname "$SCRIPT_PATH" )

participants=("004" "003" "00001" "0001")
for participant in "${participants[@]}"
do
    cd "$REPO_ROOT" && python3 src/tools/project_data.py -p "$participant"
done
