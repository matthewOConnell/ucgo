#!/usr/bin/env bash
./build/vul/ucgo -g 10 10 10 | grep "L2" | tee output
diff -s output regression_2nd_10x10x10.txt

