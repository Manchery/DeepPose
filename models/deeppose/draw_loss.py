import matplotlib.pyplot as plt
import os

os.system("grep Iteration ./examples/deeppose/deeppose.log | grep loss | awk '{print $6,$13}' | sed 's/\,//' > ./examples/deeppose/loss.data")

x=[]
y=[]
with open('./examples/deeppose/loss.data') as f:
    for line in f:
        sps=line.split()
        x.append(int(sps[0]))
        y.append(float(sps[1]))
plt.plot(x,y)
plt.show()
