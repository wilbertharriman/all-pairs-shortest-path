#!/bin/sh

INPUT_FILE=cases/c05.1
OUTPUT_FILE=test.out

srun -N1 -n1 --gres=gpu:1 cuda-memcheck ./a.out $INPUT_FILE $OUTPUT_FILE
