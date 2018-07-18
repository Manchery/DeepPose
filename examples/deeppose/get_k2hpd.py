import sys 
import numpy as np
import lmdb
#import caffe
import sys
from numpy import array

from globalvar import *

ls_path=DSPATH+"stage1/"
im_path=IMPATH
ex_path="./examples/deeppose/"

def getdata(mark):
    
    fin=open(ls_path+mark+"_set.txt","r")
    
    imglist = []
    
    for line in fin.readlines():
        data=line.split()
        imglist.append(im_path+data[0])
        x=[]
        for t in data[1].split(',')[0:30]:
            x.append(float(eval(t)))
        x=np.array(x)
        x[0::2]=x[0::2]-0.5*SIZE[0]
        x[1::2]=x[1::2]-0.5*SIZE[1]
        for t in x:
            imglist[-1]+=" "+str(t)

    fin.close()
    return imglist

def putdata(mark,data):
    fout=open(ex_path+mark+"_images.txt","w")
    for line in data:
        fout.write(line+"\n")
    fout.close()

train=getdata("train")
test=getdata("test")

print "Train: ",len(train)
print "Test: ",len(test)

putdata("train",train)
putdata("test",test)

