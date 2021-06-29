# OpenCV program to perform Edge detection
# import libraries of python OpenCV  
# where its functionality resides 
import cv2  
import matplotlib.pyplot as plt
# np is an alias pointing to numpy library 
import numpy as np 
  
def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def region_of_interest(image,vertices): 

	mask = np.zeros_like(image) 

	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(image.shape) > 2:
		channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	  
	# Fill poly-function deals with multiple polygon 
	cv2.fillPoly(mask, vertices, ignore_mask_color)  
	  
	# Bitwise operation between canny image and mask image 
	masked_image = cv2.bitwise_and(image, mask)  
	return masked_image 

def create_coordinates(image, line_parameters): 
	slope, intercept = line_parameters 
	y1 = image.shape[0] 
	y2 = int(y1 * (3 / 5)) 
	x1 = int((y1 - intercept) / slope) 
	x2 = int((y2 - intercept) / slope) 
	return np.array([x1, y1, x2, y2]) 

def average_slope_intercept(image, lines): 
	left_fit = [] 
	right_fit = [] 
	for line in lines: 
		x1, y1, x2, y2 = line.reshape(4) 
		  
		# It will fit the polynomial and the intercept and slope 
		parameters = np.polyfit((x1, x2), (y1, y2), 1)  
		slope = parameters[0] 
		intercept = parameters[1] 
		if slope < 0: 
			left_fit.append((slope, intercept)) 
		else: 
			right_fit.append((slope, intercept)) 
			  
	left_fit_average = np.average(left_fit, axis = 0) 
	right_fit_average = np.average(right_fit, axis = 0) 
	left_line = create_coordinates(image, left_fit_average) 
	right_line = create_coordinates(image, right_fit_average) 
	return np.array([left_line, right_line]) 
	
def display_lines(image, lines): 
	line_image = np.zeros_like(image) 
	if lines is not None: 
		for x1, y1, x2, y2 in lines: 
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) 
	return line_image

# capture frames from a camera 
cap = cv2.VideoCapture("test_video.mp4") 
print(cap.isOpened())
# loop runs if capturing has been initialized 
while(1): 
  
	ret, frame = cap.read() 
  
	# converting BGR to HSV (Hue Saturation Value)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
	  
	# define range of yellow color in HSV 
	lower_yellow = np.array([20,100,100], dtype = "uint8") 
	upper_yellow = np.array([30,255,255], dtype= "uint8") 
	  
	mask = cv2.inRange(hsv, lower_yellow, upper_yellow) 
  
	# Bitwise-AND mask and original image 
	res = cv2.bitwise_and(frame,frame, mask= mask) 
  
	
	# finds edges in the input image image and 
	# marks them in the output map edges 
	edges = cv2.Canny(frame,50,150) 

	#gaussian filtering
	blur = gaussian_blur(edges,5)
  
	#calculating the vertices for finding our region of interest
	imshape = frame.shape
	lower_left = [imshape[1]/9,imshape[0]]
	lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
	top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
	top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
	vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

	#Selecting Region of Interest 
	Cropped_Image = region_of_interest(blur,vertices) 
	
	#Hough Line Transformation
	lines = cv2.HoughLinesP(Cropped_Image, 2, np.pi / 180, 100, 
							np.array([]), minLineLength = 90, 
							maxLineGap = 6) 
	
	averaged_lines = average_slope_intercept(frame, lines) 
	line_image = display_lines(frame, averaged_lines) 
	combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) 

	cv2.imshow("Results", combo_image) 
	
	#displaying the video with region of interest
	cv2.imshow('Region of Interest',Cropped_Image)
    
    # display video after gaussian filtering
	cv2.imshow('Gaussian Filtering',blur) 
    
	# Display edges in a frame 
	cv2.imshow('Edges',edges)

	# Wait for Esc key to stop 
	k = cv2.waitKey(6) & 0xFF
	if k == 27: 
		break
  
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()