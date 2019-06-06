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

cd $(git rev-parse --show-toplevel)

LEVEL=$1
NEW_VER=$(bump2version ${LEVEL} --dry-run --list | grep new_version | awk -F= '{print $2}')
REPLACE="## In progress\n\n* undocumented\n\n## ${NEW_VER} ($(date '+%Y-%m-%d'))"

CHANGELOG=CHANGELOG.md
sed -i "s|## In progress|${REPLACE}|g" ${CHANGELOG}

git add ${CHANGELOG}
git commit -m "Bump changelog for v${NEW_VER}"

bump2version ${LEVEL}
