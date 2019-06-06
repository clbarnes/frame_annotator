#!/bin/bash

#This script lives in ./scripts, and uses bump2version
#to bump your changelog as well as the rest of your code.
#The changelog lives at ./CHANGELOG.md, and unreleased changes look like
#
# ## In progress
#
# * Some change
# * Some other change
#
# Use this script by running `bash scripts/bump_version.sh minor` (or whatever level)

set -e

LEVEL=$1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CFG=$(realpath ${DIR}/../.bumpversion.cfg)
BUMP_LIST=$(bump2version ${LEVEL} --dry-run --list --config-file ${CFG})
NEW_VER=$(echo ${BUMP_LIST} | grep new_version | awk -F= '{print $2}')
REPLACE="## In progress\n\n* undocumented\n\n## ${NEW_VER} $(date "+%Y-%m-%d")"

CHANGELOG=$(realpath ${DIR}/../CHANGELOG.md)
sed -i "s|## In progress|${REPLACE}|g" ${CHANGELOG}

git add ${CHANGELOG}
git commit -m "Bump changelog for v${NEW_VER}"

bump2version ${LEVEL} --config-file ${CFG}
