import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
import pickle
import glob

# Number of corners
x_cor = 9
y_cor = 6

objp = np.zeros((y_cor*x_cor,3), np.float32)
objp[:,:2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1,2)

# Object and image points from all the images.
objpoints = [] # 3d points
imgpoints = [] # 2d points

# Path to calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Search for chessboard corners
corners_not_found = []
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Conver to grayscale
    ret, corners = cv2.findChessboardCorners(gray, (x_cor,y_cor), None) # Find the chessboard corners
    # If found, add object and image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (x_cor,y_cor), corners, ret)
    else:
        corners_not_found.append(fname)
# Undistortion process
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
# Camera calibration given object and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

img_path = 'test_images/test3.jpg'
img = mpimg.imread(img_path)
undist = undistort(img)

offset = 350
img_size = undist.shape[0:2][::-1]
src = np.array([[596,450],[685, 450],[1105,720], [205, 720]], np.int32)
src = src.reshape((-1,1,2))
dst = np.array([[0+offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [0+offset, img_size[1]]])
dst = dst.reshape((-1,1,2))
M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
binary_warped = warp(convert_image(undist))
left_fit, right_fit, left_fitx, right_fitx, left_lane_inds,right_lane_inds, rectangles = fit_lanes(binary_warped)
curve_left = curve_radius(binary_warped, left_fitx)
curve_right = curve_radius(binary_warped, right_fitx)
center_distance(binary_warped, left_fit, right_fit)

def begin_draw_lane(img):
    undist = undistort(img)

    offset = 350
    img_size = undist.shape[0:2][::-1]
    src = np.array([[596,450],[685, 450],[1105,720], [205, 720]], np.int32)
    src = src.reshape((-1,1,2))
    dst = np.array([[0+offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [0+offset, img_size[1]]])
    dst = dst.reshape((-1,1,2))
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    Minv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    binary_warped = warp(convert_image(undist))
    left_fit, right_fit, left_fitx, right_fitx, left_lane_inds,right_lane_inds, rectangles = fit_lanes(binary_warped)
    curve_left = curve_radius(binary_warped, left_fitx)
    curve_right = curve_radius(binary_warped, right_fitx)
    center_distance(binary_warped, left_fit, right_fit)

    image_lane = draw_lane(undist, left_fit, right_fit)
    #image_plotter(image_lane, title='Image with lane highlighted ({})'.format(img_path))
    return image_lane

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def abs_sobel_thresh(image, orient='x', kernel_size=3, thresh=(0,255)):
    ori = [1, 0] if orient == 'x' else [0, 1]
    sobel = cv2.Sobel(image, cv2.CV_64F, ori[0], ori[1], ksize=kernel_size)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, kernel_size=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] =1
    return mag_binary

def dir_threshold(image, kernel_size=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] =1
    return dir_binary

def thresh(image, thresh=(0, 255)):
    thresh_binary = np.zeros_like(image)
    thresh_binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return thresh_binary

def convert_image(image):
    # gray sobel_x & direction
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray_x_binary = abs_sobel_thresh(gray, thresh=(10, 200))
    gray_dir_binary = dir_threshold(gray, thresh=(.5, 1.6))
    gray_combined = (gray_x_binary == 1) & (gray_dir_binary == 1)

    # yellow lines
    rgb_r = image[:,:,0]
    rgb_g = image[:,:,1]
    rgb_r_binary = thresh(rgb_r, (150, 255))
    rgb_g_binary = thresh(rgb_g, (150, 255))
    rgb_combined = (rgb_r_binary == 1) & (rgb_g_binary == 1)

    # Both of these are quite noisy, but are combined below in selector in a way that makes them more stable
    hls_l = hls[:,:,1]
    hls_s = hls[:,:,2]
    hls_l_binary = thresh(hls_l, (120, 255))
    hls_s_binary = thresh(hls_s, (100, 255))

    # Adds a little bit of information in highly irregularly shadowed areas like under trees
    hsv_v = hsv[:,:,2]
    hsv_v_sobel_x = abs_sobel_thresh(hsv_v, thresh=(30, 100))
    hls_s_sobel_x = abs_sobel_thresh(hls_s, thresh=(30, 100))

    selector = ((rgb_combined == 1) & (hls_l_binary == 1)) & ((hls_s_binary == 1) | (gray_combined == 1)) |\
        (((hsv_v_sobel_x == 1) | (hls_s_sobel_x == 1)))

    combined = np.zeros_like(rgb_r)
    combined[selector] = 1

    return combined

def image_plotter(image, title='', cmap=None, show_axis=False):
    f, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.set_title(title, fontsize=20)
    ax.imshow(image, cmap=cmap)
    if not show_axis:
        ax.set_axis_off()
    plt.show()

def warp(image):
     return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

def draw_lines(image, points, color=(255, 0, 0)):
    return cv2.polylines(np.copy(image), np.int32([points]), True, color, 3)

def fit_lanes(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Define empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Rectangles, returned if we want them for visalization
    rectangles = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Collect the windows
        rectangles.append(((win_xleft_low,win_y_low), (win_xleft_high,win_y_high)))
        rectangles.append(((win_xright_low,win_y_low), (win_xright_high,win_y_high)))

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, rectangles

def visualize_fit_lanes(binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    output_image = np.dstack((binary_warped, binary_warped, binary_warped))*255
    for rectangle in rectangles:
        cv2.rectangle(output_image, rectangle[0], rectangle[1], (0,255,0), 2)

    output_image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    output_image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    f, ax = plt.subplots(1, 1, figsize=(20,10))
    ax.imshow(output_image)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

def curve_radius(image, fitx):
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    radius = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return radius

def center_distance(image, left_fit, right_fit):
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    y_eval = np.max(ploty)
    image_center = image.shape[1]/2
    left_lane = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_lane = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lanes_center = (left_lane + right_lane)/2

    # Lanes are rought 600 pixels apart at the lower end of the image (meters per pixel in x dimension)
    xm_per_pix = 3.7/600
    meters_center = (image_center-lanes_center)*xm_per_pix
    return meters_center

def draw_lane(undist, left_fit, right_fit, text=True):
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img = cv2.imread('test_images/test3.jpg')
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    if text:
        curve_left = curve_radius(undist, left_fitx)
        curve_right = curve_radius(undist, right_fitx)
        curve_avg = (curve_left + curve_right)/2
        off_center = center_distance(img, left_fit, right_fit)

        radius_text = "Radius of curvature: {:.2f}m".format(curve_avg)
        center_dir = off_center < 0 and "left" or "right"
        center_text = "Vehicle is {:.2f}m {} of center".format(abs(off_center), center_dir)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255., 255., 255.)

        cv2.putText(result, radius_text, (20, 40), font, 1.5, font_color, 2, cv2.LINE_AA)
        cv2.putText(result, center_text, (20, 80), font, 1.5, font_color, 2, cv2.LINE_AA)
    return result
