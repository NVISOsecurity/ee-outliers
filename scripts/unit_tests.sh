#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Move to the project parent folder
cd $DIR/..

# Build the image
docker build -t "outliers-dev" .

# Run all unit tests
docker run -v "$PWD/defaults:/mappedvolumes/config" -i outliers-dev:latest python3 outliers.py tests --config /mappedvolumes/config/outliers.conf