#!/usr/bin/env bash 

set -e

if [ ! -d "./models/deeppose/snapshot/" ]; then mkdir ./models/deeppose/snapshot; fi

caffe train -solver ./models/deeppose/deeppose_solver.prototxt 2>&1| tee ./models/deeppose/deeppose.log
