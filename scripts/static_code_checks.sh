#!/usr/bin/env bash

# Build the image
docker build -t "outliers-dev" .

# Check for dead code
docker run -v "$PWD/defaults:/mappedvolumes/config" -i  outliers-dev:latest python3 -m vulture /app

# Check for PEP8 compliance
docker run -v "$PWD/defaults:/mappedvolumes/config" -i outliers-dev:latest flake8 /app "--ignore=E501"
