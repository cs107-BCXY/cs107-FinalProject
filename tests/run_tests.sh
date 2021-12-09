#!/usr/bin/env bash

tests=(
    tests/test_variable.py
    tests/test_elementary_functions.py
    tests/test_forward.py
)

if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    coverage run --source=src -m unittest discover --start-directory tests --pattern test_*.py
    coverage report --show-missing
else
    unit='-m unittest'
    if [[ $# -gt 0 && ${1} == 'pytest'* ]]; then
        driver="${@}"
    else
        driver="python ${@} ${unit}"
    fi
    ${driver} ${tests[@]}
fi