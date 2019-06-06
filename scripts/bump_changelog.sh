#!/bin/bash

set -e

LEVEL=$1
NEW_VER=$(bump2version ${LEVEL} --dry-run --list | grep new_version | awk -F= '{print $2}')
REPLACE="## In progress\n\n* undocumented\n\n## ${NEW_VER}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CHANGELOG=$(realpath ${DIR}/../CHANGELOG.md)
sed -i "s|## In progress|${REPLACE}|g" ${CHANGELOG}
