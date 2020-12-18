# Harrison Hidalgo
# MAE 6760
# For testing different functions.

import numpy as np
import runge_kutta
from SR_SPF_Ball import *
import math 
import matplotlib
import matplotlib.pyplot as plt
from EKF_filter import *
from SPF_Ball import *

N = 500
n_sig = 4.0
end_time = 5
time = np.linspace(0,end_time,N)

P_x = np.array([[1,0,0,0.5,0,0],[0,1,0,0,0.5,0],[0,0,1,0,0,0.5],[0.5,0,0,1,0,0],[0,0.5,0,0,1,0],[0,0,0.5,0,0,1]])
P_v = np.identity(6)*0.001
P_n = np.identity(6)*0.01

Q = np.identity(6)*0.001
R = np.identity(6)*0.01

w=np.random.normal(0,math.sqrt(P_v[1,1]),(1,N))
v=np.random.normal(0,math.sqrt(P_n[1,1]),(1,N))

for i in range(1,6):
	w = np.concatenate((w,np.random.normal(0,math.sqrt(P_v[i,i]),(1,N))))
	v = np.concatenate((v,np.random.normal(0,math.sqrt(P_n[i,i]),(1,N))))

measurement = np.array([1,1,1,1,1,1])

S_x0 = np.linalg.cholesky(P_x)
S_v0 = np.linalg.cholesky(P_v)
S_n0 = np.linalg.cholesky(P_n)

h = end_time/N

x_path = np.array([np.cos(time),time*0,time*0,-np.sin(time),time*0,time*0])

x_noisy = x_path + w
z_noisy = x_noisy + v
print(z_noisy.shape)
#print(x_path)
#plt.plot(time,z_noisy[0,:],'g')
#plt.plot(time,x_noisy[0,:],'r')
#plt.plot(time,x_path[0,:],'b')
#plt.show()
#plt.close()
x_hat_SPF = np.zeros((6,N))
x_hat_EKF = np.zeros((6,N))
cov   = np.zeros((6,N))

x_hat_SPF[:,None,0], Pk2k2_SPF = SPF_Ball(z_noisy[:,None,0],P_x,Q,R,n_sig,z_noisy[:,None,i],h)
x_hat_EKF[:,None,0],Pk2k2_EKF = EKF_filter(z_noisy[:,None,0],P_x,z_noisy[:,None,0],P_v,P_n,h,time[0])
#x_hat[:,None,0],S_xk = SR_SPF_Ball(z_noisy[:,None,0],S_x0,S_v0,S_n0,n_sig,z_noisy[:,None,0],h)
#print(x_hat[:,None,0].shape)
print('start')
for i in range(1,N):
	x_hat_EKF[:,None,i], Pk2k2_EKF = EKF_filter(x_hat_EKF[:,None,i-1],Pk2k2_EKF,z_noisy[:,None,i],P_v,P_n,h,time[i])
	#x_hat[:,None,i],S_xk = SR_SPF_Ball(x_hat[:,None,i-1],S_xk,S_v0,S_n0,n_sig,z_noisy[:,None,i],h)
	#cov[:,i] = np.diag(np.matmul(S_xk,np.transpose(S_xk)))
	x_hat_SPF[:,None,i], Pk2k2_SPF = SPF_Ball(x_hat_SPF[:,None,i-1],Pk2k2_SPF,Q,R,n_sig,z_noisy[:,None,i],h)
print('end')

plt.figure(1)
plt.plot(time,x_hat_EKF[0,:],'g')
plt.plot(time,x_hat_SPF[0,:],'b')
plt.plot(time,x_path[0,:],'r')

plt.figure(2)
plt.plot(time,x_path[0,:]-x_hat_EKF[0,:],'g')
plt.plot(time,x_path[0,:]-x_hat_SPF[0,:],'b')
plt.show()
