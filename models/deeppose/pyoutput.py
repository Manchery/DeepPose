#!/usr/bin/env python

import os
import caffe
import numpy as np
import cv2
import sys
import copy

from globalvar import *

STAGE1MODEL=sys.argv[1]

caffe.set_mode_gpu()
caffe.set_device(0)

#-----------------STAGE1--------------------#

def Buildnet():
    deploy='./models/deeppose/deeppose.prototxt'
    caffe_model=STAGE1MODEL
    MEAN_NPY_PATH = './models/deeppose/mean.npy' 
    if not os.path.exists(MEAN_NPY_PATH):
        print "Create mean.npy"
        MEAN_PROTO_PATH = './models/deeppose/train_mean.binaryproto'
        blob = caffe.proto.caffe_pb2.BlobProto()          
        data = open(MEAN_PROTO_PATH, 'rb' ).read()        
        blob.ParseFromString(data)                        
        array = np.array(caffe.io.blobproto_to_array(blob))
        mean_npy = array[0]
        np.save(MEAN_NPY_PATH ,mean_npy)
    net = caffe.Net(deploy,caffe_model,caffe.TEST)
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_mean('data', (np.load(MEAN_NPY_PATH).mean(1).mean(1))/255.)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    #transformer.set_raw_scale('data', 255.0)
    #net.blobs['data'].reshape(1,3,227,227)
    return net,transformer

def Test(img,net,transformer):
    im = caffe.io.load_image(img)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = copy.deepcopy(net.forward())
    # out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    return out['predict']

def process(predict,cols,rows):
    predict=predict.reshape((JOINTS,2))
    predict[:,0]+=0.5*cols
    predict[:,1]+=0.5*rows
    return predict

net,trans=Buildnet()

#---------------------Output--------------------#

imagelist="./models/deeppose/train_images.txt"
outputf=DSPATH+"stage1/output.txt"

total_test=0

with open(imagelist,'r') as f:
    fout=open(outputf,"w")
    for line in f.readlines():
        dat=(line.lstrip('\x00')).split()
        img=dat[0]
        label=np.zeros((JOINTS,2))
        for i in range(0,JOINTS*2):
            label[(i>>1),(i&1)]=eval(dat[i+1])
        label=process(label,SIZE[0],SIZE[1])
        predict=process(Test(img,net,trans),SIZE[0],SIZE[1])

        fout.write(img)
        for x in label:
            for t in x:
                fout.write(" %lf"%t)
        for x in predict:
            for t in x: 
                fout.write(" %lf"%t)
        fout.write("\n")
        
        total_test+=1
        if total_test%100==0:
            print "Read and Predict %d Images !"%total_test

