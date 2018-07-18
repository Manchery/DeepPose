set -e

if [ ! -d "./examples/deeppose_refine/snapshot/" ]; then mkdir ./examples/deeppose_refine/snapshot; fi

caffe train -solver ./examples/deeppose_refine/deeppose_refine_solver.prototxt 2>&1| tee ./examples/deeppose_refine/deeppose_refine.log
