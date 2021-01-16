import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from PIL import Image
import numpy as np
import time

# image = data.astronaut()
start_time = time.perf_counter()
image = Image.open("marin.jpg")
image = np.asarray(image)
image = rgb2gray(image)
image = img_as_float(image)
# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=250,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

fig, ax = plt.subplots(2, figsize=(7, 7))
# ax = axes.flatten()
#
end_time = time.perf_counter()
print(start_time - end_time)
ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

# ax[1].imshow(cv[0], cmap="gray")
# ax[1].set_axis_off()
# title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
# ax[1].set_title(title, fontsize=12)

ax[1].imshow(cv[1], cmap="gray")
ax[1].set_axis_off()
ax[1].set_title("Final Level Set", fontsize=12)
#
# ax[3].plot(cv[2])
# ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()
