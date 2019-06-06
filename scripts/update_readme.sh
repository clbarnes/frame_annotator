#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
README=$(realpath ${DIR}/../README.md)

fran --help | p2c ${README} --tgt _fran
fran-rename --help | p2c ${README} --tgt _fran_rename
