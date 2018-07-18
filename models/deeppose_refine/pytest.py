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
STAGE2MODEL=sys.argv[2]

#--------------Build STAGE1 & STAGE2-------------------#

def Buildnet1():
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


def Test1(img,net,transformer):
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
    
def Buildnet2():
    deploy='./models/deeppose_refine/deeppose_refine.prototxt'
    caffe_model=STAGE2MODEL
    MEAN_NPY_PATH = './models/deeppose_refine/mean.npy'
    if not os.path.exists(MEAN_NPY_PATH):
        MEAN_PROTO_PATH = './models/deeppose_refine/train_mean.binaryproto'
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
    transformer.set_raw_scale('data', 1/255.0)
    #net.blobs['data'].reshape(1,3,227,227)
    return net,transformer


def ccut(img,bc,bh,bw):
    x1=bc[0]-bw/2;x2=bc[0]+bw/2+1
    y1=bc[1]-bh/2;y2=bc[1]+bh/2+1
    rows,cols=img.shape[0:2]
    l,r,t,b=0,0,0,0
    if x1<0:
        l=-x1;x1=0
    if x2>cols:
        r=x2-cols;x2=cols
    if y1<0:
        t=-y1;y1=0
    if y2>rows:
        b=y2-rows;y2=rows
    img=img[y1:y2,x1:x2]
    return cv2.copyMakeBorder(img,t,b,l,r,cv2.BORDER_CONSTANT,value=[0,0,0])


def Test2(im,net,transformer,size,predict):
    for j in range(0,JOINTS):
        imc=ccut(im,np.int32(predict[j]),np.int32(size),np.int32(size))
        net.blobs['data'].data[j] = transformer.preprocess('data', imc)
    out = copy.deepcopy(net.forward())
    # out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    return out['predict']

#---------------Output-----------------#
    
img=sys.argv[3]

im1=cv2.imread(img)
rows,cols=im1.shape[0],im1.shape[1]

net1,trans1=Buildnet1()
predict=process(Test1(img,net1,trans1),cols,rows)

Draw(im1,predict)
cv2.imshow("predict",im1)
        
im2=cv2.imread(img)
diam=np.sqrt(np.sum((predict[DIAM1]-predict[DIAM2])**2))

net2,trans2=Buildnet2()

out=Test2(im2,net2,trans2,alpha*diam,predict)
predict+=out/64.0*diam*alpha

print predict

Draw(im2,predict)
cv2.imshow("refine",im2)

cv2.waitKey(0)
cv2.destroyAllWindows()
