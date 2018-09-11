## Writeup Template
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./rubric/bchess.jpg "Normal Image"
[image2]: ./rubric/achess.jpg "After Finding Chessboard Corners"
[image3]: ./rubric/test5.jpg "Before distortion"
[image4]: ./rubric/undistort.jpg "Undistorted Image"
[image5]: ./rubric/binary.jpg "Binary Image"
[image6]: ./rubric/warped.jpg "Warped Image"
[image7]: ./rubric/color_fit_lines.jpg "Equation Required"
[image8]: ./rubric/curved.jpg "After identifying lane lines"
[image9]: ./rubric/tracked.jpg "Tracked Image"
[video1]: ./output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code of the cam_cal.py 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Before Finding Chessboard Corner
![alt text][image1]
After finding chessboard corner
![alt text][image2]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
When the cv2.calibrateCamera() function executed on the chessboard example images, we got the camera calibration and distortion coefficients values. They are stored in the pickle file calibration_pickle.p. The undistort.py file retrieves the two values at the beginning. Then loops through the images in test_images folder and calls undestort method to apply the distortion correction. Following image shows the corrected image to the one shown above: 
Before
![alt text][image3]
After
![alt text][image4]
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I implemented various thresholding functions in binary.py file. Using abs_sobel_thresh function to evaluate gradients in x and y direction which will identify edges in vertical and horizontal direction respectively. The binary.py evaluates the gradient magnitude in x and y orientation and generates the binary mask image. The gradx and grady performs gradient direction calculation within the given range. Finally, the color_threshold function utilizes HSV as well as HLS color space to extract value channel and saturation channel respectively. These two channels are used to highlight features in the supplied image.

By combining various color and gradient thresholds mentioned earlier it generated a binary image. Here's an example of my output for this step.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes in a file called warped.py and the same logic is used in the video generation, which appears in lines 78 through 92 in the file video.py.  The file takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the following logic for the source and destination points in the following manner:

```python
src = np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],
                      [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
offset = img_size[0]* 0.25
dst = np.float32([[offset, 0],[img_size[0]-offset,0],[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]
The equation in the above has been transformed into the code written below.
```
left_fit = np.polyfit(res_yvals, leftx, 2)
left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
left_fitx = np.array(left_fitx, np.uint32)

right_fit = np.polyfit(res_yvals, rightx, 2)
right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
right_fitx = np.array(right_fitx, np.int32)
```

This is the result after identifying the lane lines 
![alt text][image8]
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #160 through #175 in my code in image.py and same logic has been applied to the video.py to create a video of lane line tracking. These lines implements the logic to measure the curvature radius. It also converts the pixel values to real world unit (m).

Using the following formula the radius of curvature is determined.

          (1+(2Ay+B)2)3/2​
Rcurve ​= -----------------
               ∣2A∣

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #160 through #175 in my code in 'image.py'. Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
