#!/usr/bin/env python

import cv2
import os
import sys
import numpy as np
import random
import copy
random.seed(10086)

from globalvar import *

os.system("mkdir "+DSPATH+"stage1")

train_annos="./data/K2HPD/train_annos.txt"
test_annos="./data/K2HPD/test_annos.txt"
train_set=DSPATH+"stage1/train_set.txt"
test_set=DSPATH+"stage1/test_set.txt"

labels={}

tra_imglist=[]
tra_files=[]

tes_imglist=[]
tes_files=[]

#------------------Read K2HPD dataset-----------------#

def readfile(file,imglist,files):
    fin=open(file,"r")
    for line in fin.readlines():
        dat=(line.lstrip('\x00')).split()
        img=dat[0]
        files.append(img)
        lbl=dat[1].split(',')
        label=[]
        for t in lbl:
            label.append(eval(t))
        labels[img]=label
        imglist.append(line)
    fin.close()

print "Reading K2HPD dataset ..."

readfile(train_annos,tra_imglist,tra_files)
readfile(test_annos,tes_imglist,tes_files)

#-----------------Mirror dataset-----------------#

print "Mirroring dataset ..."

impath=IMPATH

# mirroring images

cnt=0
for file in tra_files:
    name,ext=os.path.splitext(file)
    newfile=name+"m"+ext
    ###
    img=cv2.imread(impath+file)
    if type(img)==type(None):
        print "No this file",file
    img=cv2.flip(img,1)
    cv2.imwrite(impath+newfile,img)
    ###
    label=labels[file]
    lblm=[]
    for t in label:
        lblm.append(copy.deepcopy(t))
    lblm=np.array(lblm)
    lblm[0::2]=SIZE[0]-lblm[0::2]
    ###  swap right and left joints
    lblm[3<<1],lblm[6<<1]=lblm[6<<1],lblm[3<<1]
    lblm[4<<1],lblm[7<<1]=lblm[7<<1],lblm[4<<1]
    lblm[5<<1],lblm[8<<1]=lblm[8<<1],lblm[5<<1]
    lblm[9<<1],lblm[12<<1]=lblm[12<<1],lblm[9<<1]
    lblm[10<<1],lblm[13<<1]=lblm[13<<1],lblm[10<<1]
    lblm[11<<1],lblm[14<<1]=lblm[14<<1],lblm[11<<1]
    ###
    lblm[3<<1|1],lblm[6<<1|1]=lblm[6<<1|1],lblm[3<<1|1]
    lblm[4<<1|1],lblm[7<<1|1]=lblm[7<<1|1],lblm[4<<1|1]
    lblm[5<<1|1],lblm[8<<1|1]=lblm[8<<1|1],lblm[5<<1|1]
    lblm[9<<1|1],lblm[12<<1|1]=lblm[12<<1|1],lblm[9<<1|1]
    lblm[10<<1|1],lblm[13<<1|1]=lblm[13<<1|1],lblm[10<<1|1]
    lblm[11<<1|1],lblm[14<<1|1]=lblm[14<<1|1],lblm[11<<1|1]
    ###
    #print lblm
    #Draw(img,(np.array(lblm))[0:2*JOINTS].reshape(JOINTS,2))
    line=newfile
    for i in range(0,len(lblm)):
        line+=("%s%d"%((" " if (i==0) else ","),lblm[i]))
    line+="\n"
    
    tra_imglist.append(line)
    cnt+=1
    if cnt%1000==0:
        print "Mirrored %d images ."%cnt

#---------------------Output--------------------#

print "Output train/test set list ..."

def Shuffle(imglist):
    for i in range(0,5*len(imglist)):
        x=random.randint(0,len(imglist)-1)
        y=random.randint(0,len(imglist)-1)
        imglist[x],imglist[y]=imglist[y],imglist[x]

Shuffle(tra_imglist)

ftra=open(train_set,"w")
for line in tra_imglist:
    ftra.write(line)
ftra.close()

Shuffle(tes_imglist)

ftes=open(test_set,"w")
for line in tes_imglist:
    ftes.write(line)
ftes.close()



