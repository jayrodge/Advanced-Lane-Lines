import numpy as np
import cv2 
import pickle 
import glob

dist_pickle = pickle.load(open("calibration_pickle.p","rb"))
mtx=dist_pickle[0]
dist=dist_pickle[1]

def abs_sobel_thresh(img, orient='x',sobel_kernel=3,thresh=(0,255)):
    #Calculate directional gradient
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if orient=='x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient=='y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    #Applying threshold
    binary_output[(scaled_sobel>=thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1]) ] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1]) ] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

images = glob.glob('./test_images/test*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.undistort(img,mtx,dist,None,mtx)
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient= 'x', thresh=(12,255))
    grady = abs_sobel_thresh(img, orient= 'y',thresh=(25,255))
    c_binary = color_threshold(img, sthresh=(100,255),vthresh=(50,255))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1) )] = 255
    img_size = (img.shape[1],img.shape[0])
    bot_width = 0.76 #percent of bottom trapizoid height
    mid_width = 0.085
    height_pct = 0.625
    bottom_trim = 0.935
    src = np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],
                      [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
    offset = img_size[0]* 0.255
    dst = np.float32([[offset, 0],[img_size[0]-offset,0],[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])

    #perform the transform 
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    result=warped
    write_name = './test_images/warped'+str(idx)+'.jpg'
    cv2.imwrite(write_name, result)
