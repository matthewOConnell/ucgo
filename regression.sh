#!/usr/bin/env bash
./cmake-build-debug/vul/ucgo -g 10 10 10 | grep "L2" | tee output
diff -s output regression_10x10x10.txt

