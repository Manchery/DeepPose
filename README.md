# DeepPose

**NOTE**:

 - This is not an official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659)
 - This implementation was a project for my deep learning study as a beginner. Codes and performance are crude to some degree. But I do hope it is helpful.

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

## Caffe for DeepPose

For some reason, it is requested to run on a source-code-changed caffe

The main change is about `Data` layer, `EuclideanLoss` layer and `convert_imageset_multilabel.cpp`

The whole source is in the dir `caffe-deeppose` and please compile it by yourself

(see more: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

**HINT**: since data layer changed, it is normal to raise an error when you run `make runtest` testing data layer

`pycaffe` is request also if you want to test accuracy of your trained model

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
./deeppose_train.sh
```

It may take a very LONG time about 2 - 3 days on a GPU

By default, it runs on GPU. If you want to run on CPU-only, please edit `models/deeppose/deeppose_solver.prototxt` and `models/deeppose_refine/deeppose_refine_solver.prototxt` 

## Predict a Image 

For example,

```
python ./models/deeppose/pytest.py ./models/deeppose/snapshot/deeppose_iter_120000.caffemodel ./data/K2HPD/depth_data/depth_images/00000003.png
```

or

```
python ./models/deeppose_refine/pytest.py ./models/deeppose/snapshot/deeppose_iter_120000.caffemodel ./models/deeppose_refine/snapshot/deeppose_refine_iter_110000.caffemodel ./data/K2HPD/depth_data/depth_images/00000003.png
```

## Test accuracy 

For example,

```
python ./models/deeppose/pyaccuracy.py ./models/deeppose/snapshot/deeppose_iter_120000.caffemodel
```

or

```
python ./models/deeppose_refine/pyaccuracy.py ./models/deeppose/snapshot/deeppose_iter_120000.caffemodel ./models/deeppose_refine/snapshot/deeppose_refine_iter_120000.caffemodel
```

**Since huge size of test set, it may takes a LONG time. You can change the code by yourself.**

## Other Datasets

If you want to train on other dataset, please modify the file `globalvar.py`, in which you can describe about the dataset and write your own `get_stage1.py` to accord with your dataset

And you may need to modify the deploy `deeppose_refine.prototxt`. Its input size are relative to the number of joints for we use it to process all the joints of one person as a batch.  