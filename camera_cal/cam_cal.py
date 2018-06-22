import numpy as np
import cv2
import glob
import pickle

#prepare object points like 
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

#Arrays to store object points and image points form all the images
objpoints = [] #3d points in real world space
imgpoints= [] #2d points in image plane

#make a list of calibration images
images = glob.glob('./calibration*.jpg')

#Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    #If found, add object points, image points
    if ret == True:
        print('working on ', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        #Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        write_name = 'corners_found'+str(idx)+'.jpg'
        cv2.imwrite(write_name, img)

#load image for reference
img = cv2.imread('calibration2.jpg')
print(type(img))
img_size = (img.shape[1],img.shape[0])

#Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#save the camera calibration result for later use (we wont worry about rvecs/tvecs)
dist_pickle = []
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('./calibration_pickle.p',"wb"))