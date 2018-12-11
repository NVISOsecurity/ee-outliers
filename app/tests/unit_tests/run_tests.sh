#!/usr/bin/env bash
CALLING_DIR=`pwd`
TESTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$TESTS_DIR"

PYTHONPATH="../../../../" python -m unittest discover "$TESTS_DIR" "test_*.py"
