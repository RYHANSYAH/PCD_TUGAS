import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

image_path = 'C:\\Users\\Administrator\\Downloads\\jhonCena.jpg'  
image = imageio.imread(image_path)

if len(image.shape) == 3:
    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) 

sobel_x = ndimage.sobel(image, axis=0, mode='constant')  
sobel_y = ndimage.sobel(image, axis=1, mode='constant')  

edge_magnitude = np.hypot(sobel_x, sobel_y)
edge_magnitude = np.clip(edge_magnitude, 0, 255)  

threshold_value = 100  
segmented_image = edge_magnitude > threshold_value  

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Gambar Asli")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edge_magnitude, cmap='gray')
plt.title("Hasil Deteksi Tepi (Sobel)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(segmented_image, cmap='gray')
plt.title("Hasil Segmentasi (Thresholding)")
plt.axis('off')

plt.tight_layout()
plt.show()
