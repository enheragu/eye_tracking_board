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


# participants=("001"  "002"  "007"  "007_1"  "008"  "009"  "011"  "012"  "024"  "027"  "032"  "035"  "042"  "044"  "049"  "051"  "054"  "055"  "064")
participants=("009")
# participants discarded:
# · "058" - cannot detect border due to problem in one of the pieces

## MULTIPROCESSING VERSION
trap 'echo "Interrumpido. Matando procesos hijos..."; pkill -P $$; exit 1' SIGINT

NUM_PROCESSES=4
TOPIC="gaze"
OPTIONS="-t $TOPIC --slow_analysis"
printf "%s\n" "${participants[@]}" | xargs -P $NUM_PROCESSES -I {} bash -c '
    echo "[$(date +"%H:%M:%S")] [START] Procesando participante: {}"
    mkdir -p "output/'"$TOPIC"'/{}"
    cd "$EEHA_EYEBOARD_SCRIPT_PATH"
    python3 process_video.py -p "{}" -t gaze --slow_analysis > "output/'"$TOPIC"'/{}/process_log.txt" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +"%H:%M:%S")] [END] Finalizado participante: {} (OK)"
    else
        echo "[$(date +"%H:%M:%S")] [END] Finalizado participante: {} (ERROR: $status)"
    fi
'

## SINGLE PROCESS VERSION
# participants=("002")
# for participant in "${participants[@]}"
# do
#     clear && cd "$EEHA_EYEBOARD_SCRIPT_PATH" && python3 process_video.py -p "$participant" -t 'gaze' --visualization #--slow_analysis 
#     # clear && cd "$EEHA_EYEBOARD_SCRIPT_PATH" && python3 process_video.py -p "$participant" -t 'fixations' -o
# done

# zip -rv eye_ata_extracted.zip output -x "*.mp4"

