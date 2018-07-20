set -e

CAFFE=./caffe-deeppose

export PYTHONPATH=$CAFFE/python:$PYTHONPATH
export PATH=$CAFFE/build/tools:$PATH

python ./data/K2HPD/get_stage1.py

python ./models/deeppose/get_k2hpd.py
./models/deeppose/create_data_lmdb.sh
./models/deeppose/train.sh

python ./models/deeppose/pyaccuracy.py ./models/deeppose/snapshot/deeppose_iter_60000.caffemodel
python ./models/deeppose/pyoutput.py ./models/deeppose/snapshot/deeppose_iter_60000.caffemodel

python ./data/K2HPD/get_stage2.py

python ./models/deeppose_refine/get_k2hpd2.py
./models/deeppose_refine/create_data_lmdb.sh
./models/deeppose_refine/train.sh

python ./models/deeppose_refine/pyaccuracy.py ./models/deeppose/snapshot/deeppose_iter_60000.caffemodel ./models/deeppose_refine/snapshot/deeppose_refine_iter_40000.caffemodel
