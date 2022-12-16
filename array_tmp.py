import matplotlib.pyplot as plt
import numpy as np

time = np.arange(1,4)
print(time)
freq = np.arange(11,17)
print (freq)
x,y =np.meshgrid(np.arange(1,1+len(time)),np.arange(11,11+len(freq)))
print(x)
print(y)
Zxx=np.empty(shape=(len(time),len(freq)), dtype=int)
for i in range(len(time)):
    Zxx[i,:]=np.arange(i,len(freq)+i)
print(Zxx.T)

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x,y,Zxx.T)

plt.show()