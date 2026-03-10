#!/bin/bash

module load craype-accel-nvidia80
nvcc -o p1 p1.cu
nvcc -o p1b p1b.cu
nvcc -o p2 p2.cu -lcusparse

qsub job1_without_save.pbs
qsub job1b_without_save.pbs
qsub job2_without_save.pbs