import cv2
import numpy as np

def nothing_for_trackbar(nothing):
    pass

#define convolution kernels
identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gaussian_kernel1 = cv2.getGaussianKernel(3,0)
gaussian_kernel2 = cv2.getGaussianKernel(5,0)
box_kernel = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]], np.float32)/9.0

kernels=[identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]


#read in the image, make a grey scale copy
color_photo = cv2.imread('LynsPhoto.jpg')
greyscale_photo = cv2.cvtColor(color_photo, cv2.COLOR_BGR2GRAY)
##Create the UI (Window and trackbars)
cv2.namedWindow('instagramfilter')
# arguments: trackbarName, windowName, value(initial value), count(max value), onChange(event handler)
cv2.createTrackbar('Contrast', 'instagramfilter', 1, 100, nothing_for_trackbar)
cv2.createTrackbar('Brightness', 'instagramfilter', 50, 100, nothing_for_trackbar)
cv2.createTrackbar('Filter', 'instagramfilter', 0, len(kernels)-1, nothing_for_trackbar)
cv2.createTrackbar('Grayscale', 'instagramfilter', 0, 1, nothing_for_trackbar)
cv2.putText(img,'Hello World!', 
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    lineType)
count=1
while True:
    grayscale = cv2.getTrackbarPos('Grayscale', 'instagramfilter')
    contrast = cv2.getTrackbarPos('Contrast', 'instagramfilter')
    brightness = cv2.getTrackbarPos('Brightness', 'instagramfilter')
    kernel_idx =cv2.getTrackbarPos('Filter', 'instagramfilter')
    color_modified = cv2.filter2D(color_photo, -1, kernels[kernel_idx])
    gray_modified = cv2.filter2D(greyscale_photo, -1, kernels[kernel_idx])
    color_modified = cv2.addWeighted(color_modified, contrast, np.zeros_like(color_photo), 0, brightness - 50)
    gray_modified = cv2.addWeighted(gray_modified, contrast, np.zeros_like(greyscale_photo), 0, brightness - 50)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord('s'):
        if grayscale==0:
            cv2.imwrite('myphoto-{}.png'.format(count), color_modified)
        if grayscale==1:
            cv2.imwrite('myphotogray-{}.png'.format(count), gray_modified)
        count+=1

    if grayscale ==0:
        cv2.imshow('instagramfilter', color_modified)
    else:
        cv2.imshow('instagramfilter', gray_modified)
#Clean up
cv2.destroyAllWindows()
