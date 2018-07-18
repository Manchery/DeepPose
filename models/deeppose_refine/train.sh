set -e

if [ ! -d "./models/deeppose_refine/snapshot/" ]; then mkdir ./models/deeppose_refine/snapshot; fi

caffe train -solver ./models/deeppose_refine/deeppose_refine_solver.prototxt 2>&1| tee ./models/deeppose_refine/deeppose_refine.log
