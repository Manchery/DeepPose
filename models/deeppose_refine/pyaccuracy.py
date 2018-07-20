#!/usr/bin/env python

import os
import caffe
import numpy as np
import cv2
import sys
import copy
import random

from globalvar import *

STAGE1MODEL=sys.argv[1]
STAGE2MODEL=sys.argv[2]

#-----------------STAGE1&2--------------------#

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



def Test2(im,net,transformer,size,predict):
    for j in range(0,JOINTS):
        imc=ccut(im,np.int32(predict[j]),np.int32(size),np.int32(size))
        net.blobs['data'].data[j] = transformer.preprocess('data', imc)
    out = copy.deepcopy(net.forward())
    # out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    return out['predict']



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


net1,trans1=Buildnet1()
net2,trans2=Buildnet2()

#--------------Read test set----------------#

testlist="./models/deeppose/test_images.txt"

predict=[]
label=[]
total_test=0

RATE=40

with open(testlist,'r') as f:
    for line in f.readlines():
        if random.randint(1,RATE)==1:
            continue
        dat=(line.lstrip('\x00')).split()
        img=dat[0]
        label.append(np.zeros((JOINTS,2)))
        for i in range(0,JOINTS*2):
            label[-1][(i>>1),(i&1)]=eval(dat[i+1])
        label[-1]=process(label[-1],SIZE[0],SIZE[1])
        im=cv2.imread(img)
        output=process(Test1(img,net1,trans1),SIZE[0],SIZE[1])
        diam=np.sqrt(np.sum((output[DIAM1]-output[DIAM2])**2))

        out=Test2(im,net2,trans2,diam*alpha,output)
        output+=out/64.0*diam*alpha

        predict.append(output)
        total_test+=1
        if total_test%20==0:
            print "Read and Predict %d Images !"%total_test
    if total_test%20!=0:
        print "Read and Predict %d Images !"%total_test

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

        
