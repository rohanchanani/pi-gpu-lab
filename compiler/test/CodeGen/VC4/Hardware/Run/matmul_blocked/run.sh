#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/reference"
exec ./run.sh
