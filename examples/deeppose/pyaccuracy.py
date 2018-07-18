#!/usr/bin/env python

import os
import caffe
import numpy as np
import cv2
import sys
import copy

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

def process(predict,cols,rows):
    predict=predict.reshape((JOINTS,2))
    predict[:,0]+=0.5*cols
    predict[:,1]+=0.5*rows
    return predict

net,trans=Buildnet()

#---------------Read test set---------------#

testlist="./examples/deeppose/test_images.txt"

predict=[]
label=[]
total_test=0

with open(testlist,'r') as f:
    for line in f.readlines():
        dat=(line.lstrip('\x00')).split()
        img=dat[0]
        label.append(np.zeros((JOINTS,2)))
        for i in range(0,JOINTS*2):
            label[-1][(i>>1),(i&1)]=eval(dat[i+1])
        label[-1]=process(label[-1],SIZE[0],SIZE[1])
        output=process(Test(img,net,trans),SIZE[0],SIZE[1])
        predict.append(output)
        total_test+=1
        if total_test%100==0:
            print "Read %d Images !"%total_test

predict=np.array(predict)
label=np.array(label)

#--------------------Accuracy----------------#

def dist(a,b):
    return np.sqrt(np.sum((a-b)**2))

def Diam(idx):
    return dist(label[idx,DIAM1],label[idx,DIAM2])

def PCPforOne(idx):
    detected=0;total=0
    lambd=0.5;
    for i in range(0,len(LIMBS)):
        x=LIMBS[i][0];y=LIMBS[i][1]
        leng=dist(label[idx,x],label[idx,y])
        total+=1;
        if dist(predict[idx,x],label[idx,x])<=leng*lambd and dist(predict[idx,y],label[idx,y])<=leng*lambd:
            detected+=1;
    return detected,total



def PDJforOne(idx,lambd):
    detected=0;total=0
    diam=Diam(idx);
    for i in range(0,JOINTS):
        total+=1
        if dist(label[idx,i],predict[idx,i])<=diam*lambd:
            detected+=1;
    return detected,total



def LossforOne(idx):
    ret=0
    for i in range(0,JOINTS):
        ret+=(dist(label[idx,i],predict[idx,i]))**2;
    return ret



def PCP():
    detected=0;total=0
    for i in range(0,total_test):
        ret1,ret2=PCPforOne(i)
        detected+=ret1
        total+=ret2
    return float(detected)/total



def PDJ(lambd):
    detected=0;total=0
    for i in range(0,total_test):
        ret1,ret2=PDJforOne(i,lambd)
        detected+=ret1
        total+=ret2
    return float(detected)/total;


def Loss():
    ret=0;
    for i in range(0,total_test):
        ret+=LossforOne(i)
    return ret/2.0/total_test/(JOINTS*2.0)

print "Loss: ",Loss()
print "PCP: ",PCP()
for i in range(5,80,5):
    lambd=i/100.0
    print "PDJ for norm %.2lf"%lambd,": %.6lf"%PDJ(lambd)

        
