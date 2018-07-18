import sys 
import numpy as np
import lmdb
#import caffe
import sys
from numpy import array
import random
random.seed(10086)

from globalvar import *

ls_path=DSPATH+"stage2/images.txt"
im_path=DSPATH+"stage2/data/"
ex_path="./models/deeppose_refine/"

def getdata():
    Size=64
    
    fin=open(ls_path,"r")
    
    imglist = []
    
    for line in fin.readlines():
        data=line.split()
        imglist.append(data[0])
        x=eval(line.split()[1])
        y=eval(line.split()[2])
        x*=Size
        y*=Size
        imglist[-1]+=" "+str(x)+" "+str(y)

    fin.close()
    return imglist

def putdata(mark,data):
    fout=open(ex_path+mark+"_images.txt","w")
    for line in data:
        fout.write(line+"\n")
    fout.close()

data=getdata()

putdata("train",data[:-1000])
putdata("test",data[-1000:])

