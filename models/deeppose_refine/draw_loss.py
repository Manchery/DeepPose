import matplotlib.pyplot as plt
import os

os.system("grep Iteration ./examples/deeppose_refine/deeppose_refine.log | grep loss | awk '{print $6,$13}' | sed 's/\,//' > ./examples/deeppose_refine/loss.data")

x=[]
y=[]
with open('./examples/deeppose_refine/loss.data') as f:
    for line in f:
        sps=line.split()
        x.append(int(sps[0]))
        y.append(float(sps[1]))
plt.plot(x,y)
plt.show()
