
import numpy as np 
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('lena.png')

plt.imshow(img)
plt.show()

#make it B&W

bw = img.mean(axis = 2)
plt.imshow(bw, cmap = 'gray')
plt.show()

#create a Gaussian filter
W = np.zeros((20,20))
for i in xrange(20):
	for j in xrange(20):
		dist = (i-9.5)**2 + (j-9.5)**2
		W[i,j] = np.exp(-dist/ 50.);

# It takes the form how the filter looks like.
plt.imshow(W, cmap = 'gray')
plt.show()

# Now, take s the convolution

out = convolve2d(bw, W)
plt.imshow(out, cmap = 'gray')
plt.show()

print out.shape

out3 = np.zeros(img.shape)
for i in xrange(3):
	out3[:,:,i] = convolve2d(img[:,:,i], W, mode ='same')

plt.imshow(out3)
plt.show()






