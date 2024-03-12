import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy.fft as fft

def low_pass_filter(image_array, cutoff_frequency):
    # Perform Fourier Transform
    f_transform = fft.fft2(image_array)
    f_transform = fft.fftshift(f_transform)  # Shift the zero frequency component to the center

    # Create a low-pass filter mask with the same type as the image array
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2  # Center
    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 1

    # Apply the mask and inverse FFT
    filtered = f_transform * mask
    f_ishift = fft.ifftshift(filtered)
    img_back = fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize and convert to uint8
    img_back = np.clip(img_back, 0, 255)  # Clip values to the expected range
    img_back = img_back / np.max(img_back)  # Normalize the values
    img_back = (img_back * 255).astype(np.uint8)  # Convert to uint8

    return img_back

# Load the image (not in grayscale)
image_path = './female_anime_realistic_3heads__00425_.png'
image = Image.open(image_path)
image_array = np.array(image)

# Separate the channels
r_channel = image_array[:,:,0]
g_channel = image_array[:,:,1]
b_channel = image_array[:,:,2]

# Define cutoff frequency
cutoff = 30

# Apply the low-pass filter to each channel
filtered_r = low_pass_filter(r_channel, cutoff)
filtered_g = low_pass_filter(g_channel, cutoff)
filtered_b = low_pass_filter(b_channel, cutoff)

# Combine the channels back into one image
filtered_image = np.stack((filtered_r, filtered_g, filtered_b), axis=-1)

# Display the original and filtered images side by side
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122), plt.imshow(filtered_image), plt.title('Filtered Image')
plt.show()

