# Active Contour Test
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from PIL import Image
import cv2
import time

# mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
#
# if mouth_cascade.empty():
#   raise IOError('Unable to load the mouth cascade classifier xml file')
#
# # cap = cv2.VideoCapture(0)
# img = cv2.imread("face.jpg")
# # img = np.asarray(img)
# ds_factor = 0.5
#
# # ret, frame = img.read()
# img = cv2.resize(img, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
# for (x,y,w,h) in mouth_rects:
#     y = int(y - 0.15*h)
#     cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
#     break
#
# cv2.imshow('Mouth Detector', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = data.astronaut()
time_start = time.perf_counter()
img = Image.open("bare181.jpg")
img = np.asarray(img)
# img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 400)
r = 250 + 20*np.sin(s)
c = 130 + 40*np.cos(s)

init = np.array([r, c]).T

snake = active_contour(gaussian(img, 3),
                       init, alpha=0.015, beta=2, w_line=2, gamma=0.001, coordinates='rc')

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
time_end = time.perf_counter()
print(time_end - time_start)
plt.show()
