# Harrison Hidalgo and Christopher Chan
# ECE 5725 - Final Project
# 

import numpy as np
from runge_kutta import *
import Physical_Variables as p
from measurement_validation import *
import math

Pk2k2_init = np.identity(6)*0.1  # initialize the state covariance matrix
lam0 = 11.0705 # chi2.ppf(0.95,5)

def EKF_filter(x_hat_p,Pkk,measurement,Q,R,del_t,t0,num_tracks):
	G = np.identity(6)
	H = np.identity(6)
	x_hat_predict=np.zeros((6,num_tracks+1))
	Pk2k = np.zeros((6,6,num_tracks+1))
	d = np.zeros(num_tracks+1)
	# Predict
	for i in range(0,num_tracks+1):
		x_vel=np.asscalar(x_hat_p[3, i])
		y_vel=np.asscalar(x_hat_p[4, i])
		z_vel=np.asscalar(x_hat_p[5, i])
		c = p.c
		m = p.m
		F=np.array([[ 0, 0, 0,                                                                                                 1,                                                                                            0,                                                                                                 0],
		[ 0, 0, 0,                                                                                                 0,                                                                                            1,                                                                                                 0],
		[ 0, 0, 0,                                                                                                 0,                                                                                            0,                                                                                                 1],
		[ 0, 0, 0, - (c*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2))/m - (c*x_vel**2)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2)),                                     -(c*x_vel*y_vel)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2)),                                          -(c*x_vel*z_vel)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2))],
		[ 0, 0, 0,                                          -(c*x_vel*y_vel)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2)), -(c*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2) + (c*y_vel**2)/(x_vel**2 + y_vel**2 + z_vel**2)**(1/2))/m,                                          -(c*y_vel*z_vel)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2))],
		[ 0, 0, 0,                                          -(c*x_vel*z_vel)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2)),                                     -(c*y_vel*z_vel)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2)), - (c*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2))/m - (c*z_vel**2)/(m*(x_vel**2 + y_vel**2 + z_vel**2)**(1/2))]])
		#F = np.array([[np.asscalar(x_hat_p[3, i])*del_t,0,0,0,0,0],[0,np.asscalar(x_hat_p[4, i])*del_t,0,0,0,0],[0,0,np.asscalar(x_hat_p[5, i])*del_t,0,0,0],[0,0,0,0,0,0],[0,0,0,0,-p.g,0],[0,0,0,0,0,0]])
		#F = np.array([[np.asscalar(x_hat_p[3, i])*np.asscalar(del_t),0,0,0,0,0],[0,np.asscalar(x_hat_p[4, i])*np.asscalar(del_t),0,0,0,0],[0,0,np.asscalar(x_hat_p[5, i])*np.asscalar(del_t),0,0,0],[0,0,0,0,0,0],[0,0,0,0,-p.g,0],[0,0,0,0,0,0]])
		t, x_hat_predict[:,None,i] = runge_kutta(del_t,t0,x_hat_p[:,None,i])
		Pk2k[:,:,i] = np.matmul(np.matmul(F,Pkk[:,:,i]),np.transpose(F)) + np.matmul(np.matmul(G,Q),np.transpose(G))
		S = np.matmul(np.matmul(H,Pk2k[:,:,i]),np.transpose(H)) + R
		d[i] = np.matmul(np.matmul(np.transpose(measurement-np.matmul(H,x_hat_predict[:,None,i])),np.linalg.inv(S)),(measurement-np.matmul(H,x_hat_predict[:,None,i])))
	d_closest = np.asscalar(np.where(d == np.amin(d))[0])
	# Hypothesis test
	if measurement_validation(measurement,Pk2k[:,:,d_closest],del_t,lam0,R,x_hat_predict[:,None,d_closest]):
		# Kalman Gain
		K = np.matmul(np.matmul(Pk2k[:,:,d_closest],H),np.linalg.inv(np.matmul(np.matmul(H,Pk2k[:,:,d_closest]),np.transpose(H))+R))
		# Update
		x_hat = x_hat_predict + np.matmul(K,(measurement-x_hat_predict))
		Pk2k2 = np.matmul(np.matmul((np.identity(6)-np.matmul(K,H)),Pk2k[:,:,d_closest]),np.transpose(np.identity(6))-np.matmul(K,H)) + np.matmul(np.matmul(K,R),np.transpose(K))
		thread_number = d_closest
	else:
		#print('hit')
		thread_number = num_tracks+1
		#create new track
		#x_hat = x_hat_p[:,None,d_closest]
		#Pk2k2 = Pkk[:,:,d_closest]
		x_hat = measurement
		Pk2k2 = Pk2k2_init
	return x_hat, Pk2k2, thread_number
