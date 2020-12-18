# Harrison Hidalgo and Chris Chan
# ECE 5725 - Final Project
# This is the main program for our system.

######## BEFORE WHILE LOOP #########

## Import modules
import time
import RPi.GPIO as GPIO
import pygame
from pygame.locals import *
import os
import math
import Physical_Variables as p
from measurement_validation import *
import weights
import csv
from SR_SPF_Ball import SR_SPF_Ball
from picamera.array import PiRGBArray
from picamera import PiCamera
from EKF_filter import *
import cv2
import imutils
import transformations
import numpy as np

## Addresses
image_data = 'image.csv'
data = np.zeros((3,7))
with open(image_data,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

interface_states = 'interface_states.csv'

imu_data = 'imu.csv' #### make these for imus
data = np.zeros((3,4))
with open(imu_data,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

run_status = 'run.csv'

## Output to this file about the run status
run = True
with open(run_status,'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([run])

## Start other scripts
os.system('python3 camera_transmitter.py &')

## Create constants
current_time = time.time()
start_time   = time.time()
run_time     = 10
ball_radius  = 0.1397

## Ball Measurement Variables
x_hat = 0
S_x0 = np.linalg.cholesky(np.identity(6))*10
S_v0 = np.linalg.cholesky(np.identity(6))
S_n0 = np.linalg.cholesky(np.identity(6))
n_sig = 3
t_last = None
min_covariances=[10, 10, 10, 10, 10, 10]#[ball_radius/4,ball_radius/4,ball_radius/4,.076,.076,.076]
lam0 = 11.0705 # chi2.ppf(0.95,5)
N = 3 # Number of samples
n_sig = 3; # Number of sigma points
R = 1

## EKF Variables 
Q = np.identity(6)*1        # covariance of state noise
R = np.identity(6)*10       # covariance of measurement noise
Pk2k2 = np.identity(6)*100  # initialize the state covariance matrix

## SPF Variables




######## START WHILE LOOP ##########
while (current_time-start_time) < run_time and run:
    ## Update loop variable
    current_time = time.time()
    
    ## Collect measurements
    # Collect image from csv
    camera_measurements = np.zeros((6,3))
    t_camera = np.zeros((3,1))
    with open(image_data,'r') as csvfile:
        csvreader = csv.reader(csvfile)
        row_num = 0
        for row in csvreader:
            camera_measurements[:,None,row_num] = row[:6]
            t_camera[row_num] = row[6]
            row_num = row_num + 1
    
    ## Process measurements
    for i in range(0,3):
        # Check if there's a ball
        if (np.all(x_hat == 0) and not np.all(camera_measurements == 0)):
            x_hat = np.transpose(np.array([camera_measurements[i,:]]))
            t_last = t_camera[i]
            S_xk = S_x0
        elif (not np.all(camera_measurements == 0)):
            measurement = np.transpose(np.array([camera_measurements[i,:]]))
            x_hat, Pk2k2 = EKF_filter(x_hat,Pk2k2,measurement,Q,R,np.array([.23]),t_last)  ######   FIX THE DELTAT LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#######
            # x _pos,S_xk_pos = SR_SPF_Ball(x_hat,S_xk,S_v0,S_n0,n_sig,measurement,t_camera[i]-t_last) ripppp
            #print(measurement-x_pos)
            #P = np.matmul(S_xk_pos,np.transpose(S_xk_pos))
            #print(np.matmul(S_xk_pos, np.transpose(S_xk_pos)))
            if measurement_validation(measurement,Pk2k2,t_camera[i]-t_last,lam0,R,x_hat):
                t_last = t_camera[i]
                x_hat = x_pos 
                #S_xk = S_xk_pos
    
    ## Check if there's a ball
    #if (not np.all(camera_measurements == 0)):
    #    # Send measurements to interface
        '''
        with open(interface_states,'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            print((np.transpose(x_hat.flatten())))
            print(np.diag(np.matmul(S_xk,np.transpose(S_xk))))
            print(np.array([state])) 
            csvwriter.writerow(np.concatenate((np.transpose(x_hat),np.array([np.diag(np.matmul(S_xk,np.transpose(S_xk)))]),np.array([[state]]),axis=1)))
        '''
        ## Check covariances
        #if covariances_small_enough(Pk2k2,min_covariances) and (state == 1):
        #    state = 2    
