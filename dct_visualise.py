#!/usr/bin/env python3
'''
===============================================================================
Filename: dct_visualise.py
-------------------------------------------------------------------------------
Created on: 2025-06-24

License:
    MIT License
    Copyright (c) 2025 Kirill Kononenko

Note:
    This is NOT a valid JPEG encoder (a proper implementation can be found in
    jpeg_coder.py), but it provides a clear foundation for understanding
    the principles behind such compression algorithms.
===============================================================================
'''
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from imageio import imread
import sys

np_type = np.int8
quantize = True  # Enable quantization step
normalising_factor = 0.125  # Factor for DCT normalization

# The following code implements a row-column decomposition approach.
# This method leverages the separable nature of the 2D DCT formula, allowing it
# to be decomposed into two consecutive 1D DCT operations: one applied along
# the rows and another along the columns of the matrix
# derived from the initial row transformations.
# doi.org/10.1109/ECBS.2004.1316719
def block_dct_8x8(image):
    h, w = image.shape
    # Check if dimensions are multiples of 8; pad with zeros if necessary
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h != 0 or pad_w != 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        h, w = image.shape
    dct_image = np.zeros_like(image, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i + 8, j:j + 8]
            # DCT along rows and columns with normalization
            dct_block = (dct(dct(block.T, norm='ortho').T, norm='ortho') *
                         normalising_factor).astype(np_type)
            dct_image[i:i + 8, j:j + 8] = dct_block
    return dct_image.astype(np_type)


def block_idct_8x8(image):
    h, w = image.shape
    idct_image = np.zeros_like(image, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i + 8, j:j + 8]
            # IDCT along rows and columns with normalization
            idct_block = (idct(idct(block.T, norm='ortho').T,
                               norm='ortho')).astype(np_type)
            idct_image[i:i + 8, j:j + 8] = idct_block
    return idct_image


def quantize_8x8(image):
    h, w = image.shape
    quant_image = np.zeros_like(image, dtype=np_type)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i + 8, j:j + 8]
            # Simple quantization by halving values
            block_quant = (block // 2) * 2
            mask = (block_quant >= -2) & (block_quant <= 2)
            block_quant[mask] = 0
            quant_image[i:i + 8, j:j + 8] = block_quant
    return quant_image


# Load and preprocess image
img = imread(sys.argv[1], mode='F')
img = img.astype(
    np.float32) - 128  # Center pixel values around zero (ITU-T T.81¦ISO/IEC IS 10918-1)
img = img.astype(np_type)

# Processing pipeline
dct_result = block_dct_8x8(img)
quantized_result = quantize_8x8(dct_result) if quantize else dct_result
idct_result = 128 + block_idct_8x8(quantized_result)

# Visualization setup:
plt.figure(figsize=(16, 8))
plt.suptitle("DCT-Based Image Processing Pipeline", fontsize=14)

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 4, 2)
plt.title("DCT (Log Scale)")
plt.imshow(np.log(np.abs(dct_result) + 1), cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Quantized (Log Scale)")
plt.imshow(np.log(np.abs(quantized_result) + 1), cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Reconstructed")
plt.imshow(idct_result, cmap='gray')

plt.tight_layout()
plt.show()

# Save intermediate results to compressed files for analysis
np.savez_compressed('image', img)
np.savez_compressed('dct', dct_result)
np.savez_compressed('quantized_dct', quantized_result)
