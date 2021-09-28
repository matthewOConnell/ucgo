#!/usr/bin/env bash

set -u
set -e

exit_code=0
qsub -W block=true misc/k_gpu.pbs || exit_code=1
cat test.log
exit $exit_code
