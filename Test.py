# Harrison Hidalgo
# MAE 6760
# For testing the filters.

import numpy as np
import runge_kutta
from SR_SPF_Ball import *
import math 
import matplotlib
import matplotlib.pyplot as plt
from EKF_filter import *
from SPF_Ball import *
import random
import csv

N = 500
n_sig = 5.0
number_miss = 15
end_time = 5.0
time = np.linspace(0,end_time,N)

P_x = 0.1*np.transpose(np.array([[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]]))
P_v = np.identity(6)*0.01**2
P_n = np.identity(6)*0.1**2

Q = P_v
R = P_n

w_1=np.random.normal(0,math.sqrt(P_v[1,1]),(1,N))
v_1=np.random.normal(0,math.sqrt(P_n[1,1]),(1,N))
w_2=np.random.normal(0,math.sqrt(P_v[1,1]),(1,N))
v_2=np.random.normal(0,math.sqrt(P_n[1,1]),(1,N))

for i in range(1,6):
	w_1 = np.concatenate((w_1,np.random.normal(0,math.sqrt(P_v[i,i]),(1,N))))
	v_1 = np.concatenate((v_1,np.random.normal(0,math.sqrt(P_n[i,i]),(1,N))))
	w_2 = np.concatenate((w_2,np.random.normal(0,math.sqrt(P_v[i,i]),(1,N))))
	v_2 = np.concatenate((v_2,np.random.normal(0,math.sqrt(P_n[i,i]),(1,N))))

#h = end_time/N
h = 1/6

############## --- CREATE PATHS OF TWO BALLS --- #######################
x_path_1 = np.zeros((6,N))
x_path_2 = np.zeros((6,N))
x_path_1[:,None,0] = np.array([[0.0],[0.0],[0.0],[1.0],[1.0],[1.0]])
x_path_2[:,None,0] = np.array([[1.0],[-1.0],[1.0],[-1.0],[-1.0],[2.0]])
for i in range(1,N):
	ignore,x_path_1[:,None,i] = runge_kutta(h,time[i],x_path_1[:,None,i-1])
	ignore,x_path_2[:,None,i] = runge_kutta(h,time[i],x_path_2[:,None,i-1])
	x_path_1[:,None,i] = x_path_1[:,None,i] + w_1[:,None,i]
	x_path_2[:,None,i] = x_path_2[:,None,i] + w_2[:,None,i]
z_noisy_1 = x_path_1 + v_1
z_noisy_2 = x_path_2 + v_2

############## --- COMBINE PATHS INTO MEASUREMENTS --- #################
alpha = 0.5
z_noisy = np.zeros((6,N))
x_path = np.zeros((6,N))
for i in range(0,N):
	if random.random() < alpha:
		z_noisy[:,None,i] = z_noisy_1[:,None,i]
		x_path[:,None,i] = x_path_1[:,None,i]
	else:
		z_noisy[:,None,i] = z_noisy_2[:,None,i]
		x_path[:,None,i] = x_path_2[:,None,i]
x_hat_SPF = np.zeros((6,1))
x_hat_EKF = np.zeros((6,1))
num_tracks = 0
Pk2k2_EKF = np.zeros((6,6,1))
Pk2k2_SPF = np.zeros((6,6,1))
EKF_ERROR_TOTAL = 0
SPF_ERROR_TOTAL = 0
hits_1 = np.zeros(6)
hits_2 = np.zeros(6)

################## --- INITIALIZE FIGURES --- ##########################
#plt.figure(1)
#plt.suptitle('Extended Kalman Filter Error')
#plt.figure(2)
#plt.suptitle('Sigma Point Error')
plt.figure(3)
plt.suptitle('States Visualized')
plt.figure(4)
plt.suptitle('SPF and EKF Errors')

x_hat_EKF[:,None,0],Pk2k2_EKF[:,:,0],number_threads_EKF= EKF_filter(z_noisy[:,None,0],P_x,z_noisy[:,None,0],P_v,P_n,h,time[0],num_tracks)
x_hat_SPF[:,None,0],Pk2k2_SPF[:,:,0],number_threads_SPF= SPF_Ball(z_noisy[:,None,0],P_x,Q,R,n_sig,z_noisy[:,None,0],h,num_tracks)
#x_hat[:,None,0],S_xk = SR_SPF_Ball(z_noisy[:,None,0],S_x0,S_v0,S_n0,n_sig,z_noisy[:,None,0],h)
count_EKF = np.array([number_miss])
count_SPF = np.array([number_miss])

z_noisy=np.zeros((7,1))
###################### --- IMPORT DATA --- #############################
with open('camera_data_oneball.csv','r+') as csv_file:
	csv_reader = csv.reader(csv_file,delimiter=',')
	i=0
	for row in csv_reader:
		stripped = row[0].strip('\n')
		stripped = stripped.strip('[')
		stripped = stripped.strip(']')
		floats_list = []
		for item in stripped.split():
			floats_list.append(float(item))
		col_vec = np.transpose(np.array(floats_list,ndmin=2))
		z_noisy = np.concatenate((z_noisy,col_vec[:]),axis=1)
		i=i+1
miss=0
time_samples = z_noisy[6,:]
z_noisy = z_noisy[:6,:]
M,N = z_noisy.shape
################### --- START FILTERING --- ############################
print('start')
for i in range(1,N):
	if all(z_noisy[:,None,i] == np.zeros((6,1))):
		### Do nothing ###
		miss=miss+1
################## --- COMPUTE NEXT TIME STEP --- ######################
	else:
		x_hat_EKF_pos,Pk2k2_EKF_pos,thread_number_EKF = EKF_filter(x_hat_EKF,Pk2k2_EKF,z_noisy[:,None,i],P_v,P_n,h,time[i],num_tracks)
		x_hat_SPF_pos,Pk2k2_SPF_pos,thread_number_SPF= SPF_Ball(x_hat_SPF,Pk2k2_SPF,Q,R,n_sig,z_noisy[:,None,i],h,num_tracks)
		#x_hat_SPF[:,None,i], Pk2k2_SPF = SPF_Ball(x_hat_SPF[:,None,i-1],Pk2k2_SPF,Q,R,n_sig,z_noisy[:,None,i],h)
######################## --- UPDATE EKF ---#############################
		if thread_number_EKF > number_threads_EKF:
			x_hat_EKF = np.concatenate((x_hat_EKF,x_hat_EKF_pos),axis=1)
			Pk2k2_EKF = np.dstack((Pk2k2_EKF,Pk2k2_EKF_pos))
			number_threads_EKF = thread_number_EKF
			count_EKF = np.concatenate((count_EKF,np.array([number_miss])),axis=0)
		else:
			x_hat_EKF[:,None,thread_number_EKF-1] = x_hat_EKF_pos
			Pk2k2_EKF[:,:,thread_number_EKF-1] = Pk2k2_EKF_pos
			count_EKF[thread_number_EKF-1] = number_miss
######################## --- UPDATE SPF --- ############################
		if thread_number_SPF > number_threads_SPF:
			x_hat_SPF = np.concatenate((x_hat_SPF,x_hat_SPF_pos),axis=1)
			Pk2k2_SPF = np.dstack((Pk2k2_SPF,Pk2k2_SPF_pos))
			number_threads_SPF = thread_number_SPF
			count_SPF = np.concatenate((count_SPF,np.array([number_miss])),axis=0)
		else:
			x_hat_SPF[:,None,thread_number_SPF-1] = x_hat_SPF_pos
			Pk2k2_SPF[:,:,thread_number_SPF-1] = Pk2k2_SPF_pos
			count_SPF[thread_number_SPF-1] = number_miss
	
####################### --- UPDATE VARS --- ############################
		count_EKF = count_EKF-1
		count_SPF = count_SPF-1
		EKF_ERROR_TOTAL = EKF_ERROR_TOTAL + abs(x_path[:,None,i]-x_hat_EKF[:,None,thread_number_EKF-1])
		SPF_ERROR_TOTAL = SPF_ERROR_TOTAL + abs(x_path[:,None,i]-x_hat_SPF[:,None,thread_number_SPF-1])
###################### --- PLOTTING --- ################################
		plt.figure(3)
		for k in range(0,6):
			plt.subplot(3,2,k+1)
			#plt.plot(time[i],x_hat_EKF[k,thread_number_EKF-1],'o',color='r')
			#plt.plot(time[i],x_hat_SPF[k,thread_number_SPF-1],'o',color='b')
			plt.plot(time_samples[i],x_hat_EKF[k,thread_number_EKF-1],'o',color='r')
			plt.plot(time_samples[i],x_hat_SPF[k,thread_number_SPF-1],'o',color='b')
		#plt.figure(1)
		#for k in range(0,6):
		#	plt.subplot(3,2,k+1)
		#	plt.plot(time[i],x_path[k,i]-x_hat_EKF[k,thread_number_EKF-1],'k.',markersize=5)
		#	plt.plot(time[i],2*np.sqrt(np.mean(Pk2k2_EKF,axis=2)[k][k]),'r.',markersize=5)
		#	plt.plot(time[i],-2*np.sqrt(np.mean(Pk2k2_EKF,axis=2)[k][k]),'r.',markersize=5)
		#	if abs(x_path[k,i]-x_hat_EKF[k,thread_number_EKF-1]) < 2*np.sqrt(np.mean(Pk2k2_EKF,axis=2)[k][k]):
		#		hits_1[k] = hits_1[k] + 1
		#plt.figure(2)
		#for k in range(0,6):
		#	plt.subplot(3,2,k+1)
		#	plt.plot(time[i],x_path[k,i]-x_hat_SPF[k,thread_number_SPF-1],'k.',markersize=5)
		#	plt.plot(time[i],2*np.sqrt(np.mean(Pk2k2_SPF,axis=2)[k][k]),'r.',markersize=5)
		#	plt.plot(time[i],-2*np.sqrt(np.mean(Pk2k2_SPF,axis=2)[k][k]),'r.',markersize=5)
		#	if abs(x_path[k,i]-x_hat_SPF[k,thread_number_SPF-1]) < 2*np.sqrt(np.mean(Pk2k2_SPF,axis=2)[k][k]):
		#		hits_2[k] = hits_2[k] + 1
		plt.figure(4)
		for k in range(0,6):
			plt.figure(4)
			plt.subplot(3,2,k+1)
			plt.plot(time_samples[i],z_noisy[k,i]-x_hat_EKF[k,thread_number_EKF-1],'r.')
			plt.plot(time_samples[i],2*np.sqrt(np.mean(Pk2k2_EKF,axis=2)[k][k]),'y.',markersize=5)
			plt.plot(time_samples[i],-2*np.sqrt(np.mean(Pk2k2_EKF,axis=2)[k][k]),'y.',markersize=5)
			plt.plot(time_samples[i],z_noisy[k,i]-x_hat_SPF[k,thread_number_SPF-1],'b.')
			plt.plot(time_samples[i],2*np.sqrt(np.mean(Pk2k2_SPF,axis=2)[k][k]),'c.',markersize=5)
			plt.plot(time_samples[i],-2*np.sqrt(np.mean(Pk2k2_SPF,axis=2)[k][k]),'c.',markersize=5)
################# --- REMOVE STALE STATES --- ##########################
		j=0
		while j < len(count_EKF):
			if count_EKF[j] < 0:
				x_hat_EKF = np.delete(x_hat_EKF,j,axis=1)
				Pk2k2_EKF = np.delete(Pk2k2_EKF,j,axis=2)
				count_EKF = np.delete(count_EKF,j)
				number_threads_EKF = thread_number_EKF-1
			else:
				j=j+1
		j=0
		while j < len(count_SPF):
			if count_SPF[j] < 0:
				x_hat_SPF = np.delete(x_hat_SPF,j,axis=1)
				Pk2k2_SPF = np.delete(Pk2k2_SPF,j,axis=2)
				count_SPF = np.delete(count_SPF,j)
				number_threads_SPF = thread_number_SPF-1
			else:
				j=j+1
print('number_threads_EKF')
print(number_threads_EKF)
print('number_threads_SPF')
print(number_threads_SPF)
print('end')
plt.figure(3)
for i in range(1,7):
	plt.subplot(3,2,i)
	#plt.plot(time,x_path[i-1,:],'g.')
	for j in range(0,N):
		if z_noisy[i-1,j] != 0:
			plt.plot(time_samples[j],z_noisy[i-1,j],'g.')

plt.subplot(3,2,1)
plt.ylabel('Magnitude')
plt.subplot(3,2,3)
plt.ylabel('Magnitude')
plt.subplot(3,2,5)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.subplot(3,2,6)
plt.xlabel('Time')

#plt.figure(1)
#plt.subplot(3,2,1)
#plt.ylabel('Error')
#plt.subplot(3,2,3)
#plt.ylabel('Error')
#plt.subplot(3,2,5)
#plt.xlabel('Time')
#plt.ylabel('Error')
#plt.subplot(3,2,6)
#plt.xlabel('Time')
#plt.figure(2)
#plt.subplot(3,2,1)
#plt.ylabel('Error')
#axes = plt.gca()
#axes.set_xlim([0,5])
#axes.set_ylim([-0.5,0.5])
#plt.subplot(3,2,2)
#axes = plt.gca()
#axes.set_xlim([0,end_time])
#axes.set_ylim([-0.5,0.5])
#plt.subplot(3,2,3)
#plt.ylabel('Error')
#axes = plt.gca()
#axes.set_xlim([0,end_time])
#axes.set_ylim([-0.5,0.5])
#plt.subplot(3,2,4)
#axes = plt.gca()
#axes.set_xlim([0,end_time])
#axes.set_ylim([-0.5,0.5])
#plt.subplot(3,2,5)
#plt.xlabel('Time')
#plt.ylabel('Error')
#axes = plt.gca()
#axes.set_xlim([0,end_time])
#axes.set_ylim([-0.5,0.5])
#plt.subplot(3,2,6)
#plt.xlabel('Time')
#axes = plt.gca()
#axes.set_xlim([0,end_time])
#axes.set_ylim([-0.5,0.5])
plt.figure(4)

for i in range(1,7):
	plt.subplot(3,2,i)
	axes = plt.gca()
	axes.set_ylim([-.8,.8])
plt.subplot(3,2,1)
plt.ylabel('Error')
plt.subplot(3,2,3)
plt.ylabel('Error')
plt.subplot(3,2,5)
plt.xlabel('Time')
plt.ylabel('Error')
plt.subplot(3,2,6)
plt.xlabel('Time')

print('EKF success')
print(hits_1)
print('SPF success')
print(hits_2)

print('EKF ERROR AVERAGE')
print(EKF_ERROR_TOTAL/N)

print('SPF ERROR AVERAGE')
print(SPF_ERROR_TOTAL/N)




plt.show()
