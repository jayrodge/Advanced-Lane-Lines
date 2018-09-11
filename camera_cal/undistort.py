import numpy as np
import cv2 
import pickle 
import glob

dist_pickle = pickle.load(open("calibration_pickle.p","rb"))
mtx=dist_pickle[0]
dist=dist_pickle[1]



images = glob.glob('./camera_cal/calibration*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.undistort(img,mtx,dist,None,mtx)
    result=img
    write_name = './camera_cal/undistorted'+str(idx)+'.jpg'
    cv2.imwrite(write_name, result)

