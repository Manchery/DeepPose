#!/usr/bin/env bash 

set -e

if [ ! -d "./examples/deeppose/snapshot/" ]; then mkdir ./examples/deeppose/snapshot; fi

caffe train -solver ./examples/deeppose/deeppose_solver.prototxt 2>&1| tee ./examples/deeppose/deeppose.log
