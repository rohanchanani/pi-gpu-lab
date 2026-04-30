#!/usr/bin/env bash
set -euo pipefail

CC_CMD="${CC:-cc}"

"$CC_CMD" -std=c99 -O2 -Wall -Wextra -pedantic \
  maxpool2d_2x2_harness.c maxpool2d_2x2_launch.c \
  -o maxpool2d_2x2_harness_host
./maxpool2d_2x2_harness_host
