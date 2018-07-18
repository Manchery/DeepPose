set -e

CAFFE=~/mancherywu/caffe

export PYTHONPATH=$CAFFE/python:$PYTHONPATH
export PATH=$CAFFE/build/tools:$PATH

python ./data/K2HPD/get_stage1.py

python ./examples/deeppose/get_k2hpd.py
./examples/deeppose/create_data_lmdb.sh
./examples/deeppose/train.sh
#python ./examples/deeppose/pyaccuracy.py ./models/deeppose.caffemodel
#python ./examples/deeppose/pyoutput.py ./models/deeppose.caffemodel

python ./examples/deeppose/pyaccuracy.py ./examples/deeppose/snapshot/deeppose_iter_120000.caffemodel
python ./examples/deeppose/pyoutput.py ./examples/deeppose/snapshot/deeppose_iter_120000.caffemodel

python ./data/K2HPD/get_stage2.py

python ./examples/deeppose_refine/get_k2hpd2.py
./examples/deeppose_refine/create_data_lmdb.sh
./examples/deeppose_refine/train.sh
#python ./examples/deeppose_refine/pyaccuracy.py ./models/deeppose.caffemodel ./models/deeppose_refine.caffemodel
python ./examples/deeppose_refine/pyaccuracy.py ./examples/deeppose/snapshot/deeppose_iter_120000.caffemodel ./examples/deeppose_refine/snapshot/deeppose_refine_iter_120000.caffemodel
