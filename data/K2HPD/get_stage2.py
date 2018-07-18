#!/usr/bin/env python

import os
import scipy.io
import numpy as np
import cv2
import random
random.seed(10086)

from globalvar import *

os.system("mkdir "+DSPATH+"stage2")

pout_path=DSPATH+"stage1/output.txt"
pdat_path=IMPATH

ndat_path=DSPATH+"stage2/data/"
ntra_path=DSPATH+"stage2/images.txt"

os.system("rm -rf "+ndat_path)
os.system("rm -f "+ntra_path)

os.system("mkdir "+ndat_path)
os.system("touch "+ntra_path)

imgs=[]
pilist=[]
plabel=[]            
ppredi=[]

#----------------------Read output of previous stage---------------#

with open(pout_path,"r") as f:
    cnt=0
    for line in f.readlines():
        dat=(line.lstrip('\x00')).split()
        #print dat
        pilist.append(dat[0].split('/')[-1])
        imgs.append(dat[0])
        plabel.append(np.zeros((JOINTS,2)))
        ppredi.append(np.zeros((JOINTS,2)))
        for i in range(0,JOINTS*2):
            plabel[-1][(i>>1),(i&1)]=dat[i+1]
            ppredi[-1][(i>>1),(i&1)]=dat[i+JOINTS*2+1]
        cnt+=1
        if cnt%10000==0:
            print "Read %d Images !"%cnt
            
num=cnt

pilist=np.array(pilist)
plabel=np.array(plabel)
ppredi=np.array(ppredi)

#---------------------Cut with bounding box------------------------#

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

ftra=open(ntra_path,"w")

diam=0

minsize=1e5
maxsize=0
meansize=0
rate=0
tot=0

for i in range(0,num):
    diam=np.sqrt(np.sum((plabel[i,DIAM1]-plabel[i,DIAM2])**2))
    maxsize=max(maxsize,diam*alpha)
    minsize=min(minsize,diam*alpha)
    meansize=meansize+diam*alpha
    for j in range(0,JOINTS):
        f=0
        if random.randint(1,JOINTS)<=1:
            f=ftra
        else:
            continue
        img=cv2.imread(imgs[i])
        img=ccut(img,np.int32(ppredi[i,j]),np.int32(diam*alpha),np.int32(diam*alpha))

        cv2.imwrite(ndat_path+pilist[i].split('.')[0]+(".%d.jpg"%j),img)

        f.write(ndat_path+pilist[i].split('.')[0]+(".%d.jpg"%j))

        label_=(plabel[i,j]-ppredi[i,j])/(diam*alpha)

        tot+=1
        if label_[0]>=-0.5 and label_[0]<=0.5 and label_[1]>=-0.5 and label_[1]<=0.5:
            rate+=1

        f.write(" %lf %lf\n"%(label_[0],label_[1]))
    if i%1000==0:
        print "Processed %d Images."%i

#------------------Output statistics-------------------------#

meansize/=num
rate/=float(tot)
print "max of size :",maxsize
print "min of size :",minsize
print "mean of size:",meansize
print "included rate:",rate
