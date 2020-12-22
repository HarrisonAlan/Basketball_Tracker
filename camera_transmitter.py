# Harrison Hidalgo
# ECE 5725 - Final Project
# This program captures images. The images are then run through a 95% 
# hypothesis test. If they pass the test they are then stored in csv 
# files where they can be used as data samples.

import csv
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import imutils
from EKF_filter import *
from measurement_validation import *
from SPF_Ball import *

## Ball Measurement Variables
x_hat = 0
n_sig = 5.0  # Number of sigma points
t_last = None
lam0 = 11.0705 # chi2.ppf(0.95,5)
LIMIT  = 50;
time_start = time.time()

## EKF Variables 
Q = np.identity(6)*0.01      # covariance of process noise
R = np.identity(6)*0.1       # covariance of measurement noise
Pk2k2 = np.zeros((6,6,1))
Pk2k2[:,:,0] = np.identity(6)*100  # initialize the state covariance matrix

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
framesToPlot = 8

# define the lower and upper boundaries of the ball in the HSV color space
colorLower = (171, 174, 70)
colorUpper = (179, 255, 255)

# keep track of bounding box locations in a dictionary
ball_dict = {'location': [], 'time': [], 'velocity': [[0, 0, 0]], 'distance':[[0, 0, 0]]}
ballDetected = False

# parameters
r_ball = 139.7 # radius of ball (mm)
f_lens = 3.04 # camera lens focal length (mm)
h_sensor = 2.76 # camera sensor h

#eight (mm)

# writing data
csv_data = 'camera_data.csv'
estimates = 'estimates.csv'
data_to_plot = 3
data = np.zeros((1,8))
data_count = 0

# tracking times
start_time = time.time()
current_time = time.time()

# loop variables
run = True
start_time = time.time()
current_time = time.time()
end_time = 30

############## GET IMAGE 1 ##############
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    ballDetected = False
    ### *** RECOGNIZE AND RECORD LOCATION OF OBJECT *** ###
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    # Resize the frame, blur it, and convert to HSV color space
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Construct a mask for the color "orange", then perform a series of
    # dilations and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # Find contours in the mask and initialize the current (x,y) center
    # of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # Only proceed if at least one contour was found
    if len(cnts) > 0:
        # Find the largest contour in the mask, then use it to compute
        # the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # Only proceed if the radius meets a minimum size
        if radius > 20:
            # draw the circle and centroid on the frame, then update the
            # list of tracked points
            ballDetected = True
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2) 
            cv2.circle(image, center, 5, (0, 0, 255), -1) 
            ball_dict['location'].append([x, y, radius])
            ball_dict['time'].append(time.time())
    # Connect last __ frames with a blue line
    if len(ball_dict['location']) >= framesToPlot:
        center_pts = np.zeros((framesToPlot, 2), np.int32)
        for i in range(framesToPlot):
            center_pts[i][0] = ball_dict['location'][len(ball_dict['location'])-i-1][0]
            center_pts[i][1] = ball_dict['location'][len(ball_dict['location'])-i-1][1]
        center_pts = center_pts.reshape((-1,1,2))
        cv2.polylines(image,[center_pts],False,(255,0,0),5)
    # Show the frame
    cv2.imshow('Frame',image)
    # Wait for key
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    #print(ball_dict)
    ### *** GET VELOCITY *** ###
    if len(ball_dict['location']) > 1 and ballDetected:
        # get measurments of last 2 frames
        [x, y, r] = [camera.resolution[0]-ball_dict['location'][-1][0], camera.resolution[1]-ball_dict['location'][-1][1], ball_dict['location'][-1][2]]
        t = ball_dict['time'][-1]
        [x0, y0, r0] = [camera.resolution[0]-ball_dict['location'][-2][0], camera.resolution[1]-ball_dict['location'][-2][1], ball_dict['location'][-2][2]]
        t0 = ball_dict['time'][-2]
        # calculate x
        x_m = x*r_ball/r # (mm)
        x0_m = x0*r_ball/r0 # (mm)
        v_x = (x_m-x0_m)/((t-t0)*1000) # (m/s)
        # calculate y
        y_m = y*r_ball/r # (mm)
        y0_m = y0*r_ball/r0 # (mm)
        v_y = (y_m-y0_m)/((t-t0)*1000) # (m/s)
        # calculate z
        z_m = (f_lens*r_ball*camera.resolution[1])/(r*h_sensor*1000) # (m)
        z0_m = (f_lens*r_ball*camera.resolution[1])/(r0*h_sensor*1000) # (m)
        v_z = (z0_m-z_m)/(t-t0) # (m/s)
        ball_dict['distance'].append([x_m/1000, y_m/1000, z_m])
        ball_dict['velocity'].append([v_x, v_y, v_z])
        
    ### *** WRITE TO CSV FILE *** ###
    if ballDetected:
        csv_x = ball_dict['distance'][-1][0]
        csv_y = ball_dict['distance'][-1][1]
        csv_z = -ball_dict['distance'][-1][2]
        csv_vx = ball_dict['velocity'][-1][0]
        csv_vy = -ball_dict['velocity'][-1][1]
        csv_vz = -ball_dict['velocity'][-1][2]
        csv_time = ball_dict['time'][-1]
        data[data_count,None,:] = np.array([csv_x, csv_y, csv_z, csv_vx, csv_vy, csv_vz, csv_time-start_time,ballDetected])
    else:
        data[data_count,:] = np.array([0, 0, 0, 0, 0, 0, time.time()-start_time, ballDetected])
    print(data[data_count,6])
    ### *** CONDITION TO END FOR LOOP *** ###
    current_time = time.time()
    if not run or (current_time-start_time) >= end_time:
        break
    print(data)
    
    
    with open(csv_data,'a') as csvfile:
            # What is written:
            #   x,y,z,x_d,y_d,z_d,t
            data_writer = csv.writer(csvfile)
            data_writer.writerow(data[0,None,0:7])
            
    ## Process measurements
    # Check if there's a ball
    #if (np.all(x_hat == 0) and (data[0,7] == True)):
    #    x_hat = np.transpose([data[0,0:6]])
    #    t_last = data[0,None,6]-time_start
    #    number_threads = 0
    #    hits = np.array([LIMIT])
    #elif data[0,7]:
    #    measurement = np.transpose(data[0,None,0:6])
    #    x_pos, Pk2k2_pos,thread_number = EKF_filter(x_hat,Pk2k2,measurement,Q,R,data[0,None,6]-t_last-start_time,t_last,number_threads) 
    #    #x_pos,Pk2k2_pos,thread_number= SPF_Ball(x_hat,Pk2k2,Q,R,n_sig,measurement,data[0,None,6]-t_last)
    #    t_last = data[0,None,6]-time_start
    #    if thread_number > number_threads:
    #        x_hat = np.concatenate((x_hat,x_pos),axis=1)
    #        print(Pk2k2.shape)
    #        print(np.expand_dims(Pk2k2_pos,axis=(6,6,1)))
    #        Pk2k2 = np.dstack((Pk2k2,Pk2k2_pos))
    #        print('fuck')
    #        print(hits)
    #        hits = np.concatenate((hits,np.array([LIMIT])),axis=0)
    #        print(hits)
    #    else:
    #        x_hat[:,None,thread_number] = x_pos 
    #        print(Pk2k2)
    #        Pk2k2[:,:,thread_number] = Pk2k2_pos
    #        print(Pk2k2)
    #        hits[thread_number] = LIMIT
    #    with open(csv_data,'a') as csvfile:
    #        # What is written:
    #        #   x,y,z,x_d,y_d,z_d,t
    #        data_writer = csv.writer(csvfile)
    #        data_writer.writerow(data[0,None,0:7])
    #    with open(estimates,'a') as csvfile:
    #        # What is written:
    #        #   x,y,z,x_d,y_d,z_d,t,thread_number
    #        data_writer = csv.writer(csvfile)
    #        data_writer.writerow(np.concatenate((np.transpose(x_hat[:,thread_number]),t_last),axis=0))
    #if 'hits' in dir():
    #    print(hits)
    #    for i in range(0,len(hits)):
    #        hits[i] = hits[i]-1
    #        if hits[i]<0:
    #            x_hat = np.delete(x_hat,i,1)
    #            Pk2k2 = np.delete(Pk2k2,i,2)
    #            hits.remove(i)
    #            number_threads = number_threads-1
