#!/usr/bin/env python

import os
import caffe
import numpy as np
import cv2
import sys
import copy
import random
random.seed(10086)

from globalvar import * 

STAGE1MODEL=sys.argv[1]

#-----------------STAGE1--------------------#

def Buildnet():
    deploy='./examples/deeppose/deeppose.prototxt'
    caffe_model=STAGE1MODEL
    MEAN_NPY_PATH = './examples/deeppose/mean.npy' 
    if not os.path.exists(MEAN_NPY_PATH):
        print "Create mean.npy"
        MEAN_PROTO_PATH = './examples/deeppose/train_mean.binaryproto'
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

net,trans=Buildnet()

#--------------Predict and Draw--------------#

img=sys.argv[2]

im=cv2.imread(img)
rows,cols=im.shape[0],im.shape[1]

predict=Test(img,net,trans).reshape((JOINTS,2))
predict[:,0]+=0.5*cols
predict[:,1]+=0.5*rows

print predict

def Draw(im,predict):
    joi=[]
    for x in np.int32(predict):
        joi.append(tuple(x))

    clr=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0)]
    for i in range(0,len(LIMBS)):
        x=LIMBS[i][0];y=LIMBS[i][1]
        cv2.line(im,joi[x],joi[y],clr[random.randint(0,len(clr)-1)])

    '''    
    cv2.line(im,joi[0],joi[1],(255,255,0))
    cv2.line(im,joi[1],joi[2],(0,0,255))
    cv2.line(im,joi[3],joi[4],(255,255,0))
    cv2.line(im,joi[4],joi[5],(0,0,255))
    cv2.line(im,joi[6],joi[7],(0,255,0))
    cv2.line(im,joi[7],joi[8],(255,0,0))
    cv2.line(im,joi[9],joi[10],(0,255,0))
    cv2.line(im,joi[10],joi[11],(255,0,0))
    cv2.line(im,joi[12],joi[13],(255,0,255))
    cv2.line(im,joi[13],joi[14],(255,0,255))
    '''

Draw(im,predict)

cv2.imshow("predict",im)
cv2.waitKey(0)
cv2.destroyAllWindows()
