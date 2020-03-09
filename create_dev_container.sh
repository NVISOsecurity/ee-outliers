#!/bin/bash
TEMP_FOLDER=/tmp/outliers_dev
TARGET_HOST=""
SLAVE_HOSTNAME=""

function display_usage() { 
	echo "Usage: ./create_dev_container.sh --host <hostname>"
    echo -e "\t<hostname>\tThe SSH host to deploy the dev container on"
    echo -e "\t<slavename>\tThe slave hostname to set in the container, this matches the hostname configured in the master" 
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

rsync -rav $(pwd) --exclude virtualenv --exclude .git --exclude venv $TARGET_HOST:$TEMP_FOLDER --delete
ssh $TARGET_HOST "cd $TEMP_FOLDER/ee-outliers;docker-compose build;docker stop outliers;docker-compose up -d --force-recreate;docker logs outliers_dev --follow"
