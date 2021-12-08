#!/usr/bin/env bash

tests=(
    tests/test_variable.py
    tests/test_elementary_functions.py
    tests/test_forward.py
)

unit='-m unittest'
if [[ $# -gt 0 && ${1} == 'pytest'* ]]; then
    driver="${@}"
else
    driver="python ${@} ${unit}"
fi

${driver} ${tests[@]}