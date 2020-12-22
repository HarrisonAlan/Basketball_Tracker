# Harrison Hidalgo
# MAE 6760 - Final Project 
# Square-Root Sigma Point Kalman Filter Ball

### Done before: initialize

import numpy as np
from ball_calc import *
from runge_kutta import *
import weights
import math
from measurement_validation import *

Pk2k2_init = np.identity(6)*0.1  # initialize the state covariance matrix
lam0 = 11.0705 # chi2.ppf(0.95,5)
nx = 6
ensp = np.ones((1,nx*2+1));

def SPF_Ball(x_0,Pkk,Q,R,n_sig,measurement,dt,num_tracks):
  H = np.identity(6)
  G = np.identity(6)
  x_hat_predict=np.zeros((6,num_tracks+1))
  Pk2k1 = np.zeros((6,6,num_tracks+1))
  d = np.zeros(num_tracks+1)
  for i in range(0,num_tracks+1):
    ## Create sigma points
    S_kk = np.linalg.cholesky(Pkk[:,:,i])
    sigma_points=np.matmul(x_0[:,None,i],ensp)+np.concatenate((np.zeros((nx,1)),n_sig*S_kk,-n_sig*S_kk),axis=1);
    ## Predict
    X_p1 = np.zeros((nx,nx*2+1))
    time,X_p1[:,None,0]=runge_kutta(dt,0,sigma_points[:,None,0])
    x_p1=X_p1[:,None,0]*weights.wm0;
    for j in range(0,nx):
      time,X_p1[:,None,j+1] = runge_kutta(dt,0,sigma_points[:,None,1+j]);
      time,X_p1[:,None,j+1+nx] = runge_kutta(dt,0,sigma_points[:,None,1+j+nx]);
      x_p1 = weights.wm*(X_p1[:,None,j+1]+X_p1[:,None,j+1+nx])+x_p1;
    x_hat_predict[:,None,i] = x_p1
    Pk2k=weights.wc0*np.matmul(X_p1[:,None,0]-x_p1,np.transpose(X_p1[:,None,0]-x_p1))
    for j in range(0,2*nx):
      Pk2k = weights.wc*np.matmul(X_p1[:,None,j+1]-x_p1,np.transpose(X_p1[:,None,j+1]-x_p1))+Pk2k
    Pk2k1[:,:,i] = Pk2k + np.matmul(np.matmul(G,Q),np.transpose(G))
    S = np.matmul(np.matmul(H,Pk2k1[:,:,i]),np.transpose(H)) + R
    d[i]=np.matmul(np.matmul(np.transpose(measurement-np.matmul(H,x_hat_predict[:,None,i])),np.linalg.inv(S)),(measurement-np.matmul(H,x_hat_predict[:,None,i])))
  d_closest = np.asscalar(np.where(d == np.amin(d))[0])
  # Hypothesis test
  if measurement_validation(measurement,Pk2k1[:,:,d_closest],dt,lam0,R,x_hat_predict[:,None,d_closest]):
    x_p1 = x_hat_predict[:,None,d_closest]
    Pk2k1 = Pk2k1[:,:,d_closest]
    #### Update
    Sk2k1 = np.linalg.cholesky(Pk2k1)
    X_p2 = np.zeros((nx,nx*2+1))
    X_p2[:,None,0]=x_p1;
    ## Re create sigma points
    X_p2=np.matmul(x_p1,ensp)+np.concatenate((np.zeros((nx,1)),n_sig*Sk2k1,-n_sig*Sk2k1),axis=1);
    Z_k2=np.zeros((nx,2*nx+1));
    Z_k2[:,None,0] = np.matmul(H,X_p2[:,None,0])
    z_k2 = Z_k2[:,None,0]*weights.wm0
    for i in range(1,2*nx+1):
      Z_k2[:,None,i]= np.matmul(H,X_p2[:,None,i])
      z_k2=z_k2+Z_k2[:,None,i]*weights.wm;
    Pxz = weights.wc0*np.matmul(X_p2[:,None,0]-x_p1,np.transpose(Z_k2[:,None,0]-z_k2))
    Pzz = weights.wc0*np.matmul((Z_k2[:,None,0]-z_k2),np.transpose(Z_k2[:,None,0]-z_k2))+R
    for i in range(0,2*nx):
      Pxz = Pxz + weights.wc*np.matmul(X_p2[:,None,i+1]-x_p1,np.transpose(Z_k2[:,None,i+1]-z_k2))
      Pzz = Pzz + weights.wc*np.matmul(Z_k2[:,None,i+1]-z_k2,np.transpose(Z_k2[:,None,i+1]-z_k2))
    Kalman_Gain=np.matmul(Pxz,np.linalg.inv(Pzz));
    innovation=measurement-z_k2
    x_hat=x_p1+np.matmul(Kalman_Gain,innovation)
    Pk2k2 = Pk2k1 - np.matmul(np.matmul(Pxz,np.linalg.inv(Pzz)),np.transpose(Pxz))
    thread_number = d_closest
  else:
    #print('hit')
    thread_number = num_tracks+1
		#create new track
    #x_hat = x_0[:,None,d_closest]
    #Pk2k2 = Pkk[:,:,d_closest]
    x_hat = measurement
    Pk2k2 = Pk2k2_init
  return x_hat,Pk2k2,thread_number

def validateCovMatrix(sig):
  EPS = 10**-6
  ZERO= 10**-10
  sigma = sig
  if (is_pos_def(sigma) == False):
    w,v = np.linalg.eig(sigma)
    for n in range(0,len(w)):
      if (v[n,n] <= ZERO):
        v[n,n] = EPS
      sigma = w*v*np.transpose(w)
  return sigma
    
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
