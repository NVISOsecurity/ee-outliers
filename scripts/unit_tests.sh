#!/usr/bin/env bash

# Build the image
docker build -t "outliers-dev" .

# Run all unit tests
docker run -v "$PWD/defaults:/mappedvolumes/config" -i outliers-dev:latest python3 outliers.py tests --config /mappedvolumes/config/outliers.conf