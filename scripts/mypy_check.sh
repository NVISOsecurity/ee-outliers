#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Move to the project parent folder
cd $DIR/..

mypy app/outliers.py
