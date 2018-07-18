# DeepPose

## Introduction 

This is a implementation at [caffe](http://caffe.berkeleyvision.org/) in DeepPose proposed in [this paper](http://arxiv.org/abs/1312.4659), with [Kinect2 Human Pose Dataset (K2HPD)](http://www.sysu-hcp.net/kinect2-human-pose-dataset-k2hpd/)

Caffe is a deep learning framework

>@article{jia2014caffe,
  Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
  Journal = {arXiv preprint arXiv:1408.5093},
  Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
  Year = {2014}
} 

The K2HPD is from *Keze Wang, Shengfu Zhai, Hui Cheng, Xiaodan Liang, and Liang Lin. Human Pose Estimation from Depth Images via Inference Embedded Multi-task Learning. In Proceedings of the ACM International Conference on Multimedia (ACM MM), 2016.*

Following command should be run in the root directory of this project if not specified

## Changed caffe

For some reason, it is requested to run on a source-code-changed caffe

The main change is about `Data` layer, `EuclideanLoss` layer and `convert_imageset_multilabel.cpp`

The whole source is in the dir `caffe` and please compile it by yourself

(see more: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

**With dependencies installed**,try run these in the dir `caffe`

```
make all
make test
make runtest
```

`pycaffe` is request also if you want to test accuracy of your trained model

Run these in the dir `caffe`

```
make pycaffe
make distribute
```

## Data Preparation for K2HPD

Please download the dataset from the link by yourself and copy them to `data/K2HPD`

```
cp -r your/path/to/depth_data data/K2HPD
cp your/path/to/train_annos.txt data/K2HPD
cp your/path/to/test_annos.txt data/K2HPD
```

## Start Training 

First **make sure** there are no other caffes in your environment path and python environment path, then

```
./deeppose.sh
```

It may take a very LONG time

## Predict a Image 

For example,

```
python ./models/deeppose/pytest.py ./models/deeppose/snapshot/deeppose_iter_120000.caffemodel ./data/K2HPD/depth_data/depth_images/00000003.png
```

or

```
python ./models/deeppose_refine/pytest.py ./models/deeppose/snapshot/deeppose_iter_120000.caffemodel ./models/deeppose_refine/snapshot/deeppose_refine_iter_110000.caffemodel ./data/K2HPD/depth_data/depth_images/00000003.png
```

## Other Datasets

If you want to train on other dataset, please modify the file `globalvar.py`, in which you can describe about the dataset and write your own `data/K2HPD/stage/get_stage1.py` to accord with your dataset