#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Move to the project parent folder
cd $DIR/..

mypy --ignore-missing-imports --disallow-untyped-defs --disallow-incomplete-defs --check-untyped-defs app/outliers.py
