#!/bin/bash
TEMP_FOLDER=/tmp
TARGET_HOST=""

function display_usage() {
	echo "Usage: $0 --host <username>@<hostname>"
    echo -e "\t<hostname>\tThe SSH username and host to sync the files to"
    echo ""
    exit
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--host)
    TARGET_HOST="$2"
    shift # past argument
    shift # past value
    ;;
    --help)
    display_usage
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

if [ "$TARGET_HOST" == "" ] ; then
    display_usage
fi
fswatch -e tmp/ -o . | xargs -n 1 -I{} sh -c "rsync -rav --delete $(pwd)/requirements.txt $(pwd)/defaults $(pwd)/scripts $(pwd)/Dockerfile $(pwd)/app $TARGET_HOST:$TEMP_FOLDER/ee-outliers/;afplay /System/Library/Sounds/Pop.aiff"

