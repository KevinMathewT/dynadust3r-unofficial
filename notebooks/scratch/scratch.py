import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/scratch/km6748/data/point_odyssey/train/animal8/rgbs/rgb_00000.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.savefig('out.png', bbox_inches='tight', pad_inches=0)
