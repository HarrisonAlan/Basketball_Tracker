# Harrison Hidalgo
# MAE 6760 - Final Project 
# Square-Root Sigma Point Kalman Filter Ball

### Done before: initialize

import numpy as np
from ball_calc import *
from runge_kutta import *
import weights
import math

def SPF_Ball(x_0,Pkk,Q,R,n_sig,measurement,dt):
  # S_v0 is process noise
  # S_n0 is sensor noise
  H = np.identity(6)
  G = np.identity(6)
  S_kk = np.linalg.cholesky(Pkk)
  ## Create sigma points
  nx = len(x_0)
  ensp = np.ones((1,nx*2+1));
  sigma_points=np.matmul(x_0,ensp)+np.concatenate((np.zeros((nx,1)),n_sig*S_kk,-n_sig*S_kk),axis=1);
  ## Predict
  X_p1 = np.zeros((nx,nx*2+1))
  time,X_p1[:,None,0]=runge_kutta(dt,0,sigma_points[:,None,0])
  x_p1=X_p1[:,None,0]*weights.wm0;
  for i in range(0,nx):
    time,X_p1[:,None,i+1] = runge_kutta(dt,0,sigma_points[:,None,1+i]);
    time,X_p1[:,None,i+1+nx] = runge_kutta(dt,0,sigma_points[:,None,1+i+nx]);
    x_p1 = weights.wm*(X_p1[:,None,i+1]+X_p1[:,None,i+1+nx])+x_p1;
  Pk2k1=weights.wc0*np.matmul(X_p1[:,None,0]-x_p1,np.transpose(X_p1[:,None,0]-x_p1))
  for i in range(0,2*nx):
    Pk2k1 = weights.wc*np.matmul(X_p1[:,None,i+1]-x_p1,np.transpose(X_p1[:,None,i+1]-x_p1))+Pk2k1
  Pk2k1 = Pk2k1 + np.matmul(np.matmul(G,Q),np.transpose(G))
  
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
  return x_hat,Pk2k2

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
