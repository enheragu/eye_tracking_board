#!/usr/bin/env bash


## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
SOURCE=$(readlink "$SOURCE")
[[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
export EEHA_EYEBOARD_SCRIPTFILE_PATH=$SOURCE
export EEHA_EYEBOARD_SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )


participants=("004" "003" "00001" "0001")
for participant in "${participants[@]}"
do
    cd "$EEHA_EYEBOARD_SCRIPT_PATH" && python3 project_data.py -p "$participant"    
done