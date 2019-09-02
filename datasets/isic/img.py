import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mask = Image.open('./train_mask/ISIC_0002374_segmentation.png')
mask = np.array(mask).astype('int32')
mask[mask == 255] = -1
plt.figure('mask')
plt.imshow(mask)
plt.show()