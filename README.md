# Python JPEG Encoder

<p align="center">
     <img src="https://github.com/Kononenko-K/Python-JPEG-Encoder/blob/main/pics/header.png">
</p>

## Overview
The [first script](jpeg_coder.py) is a pure Python (+ Numpy) implementation of a JPEG 
encoder that accepts an uncompressed raster BMP RGB24 image file as input 
and performs the following steps:
- Conversion to YCbCr color space
- Splitting the image into 8x8 blocks
- Applying a fast 2-D DCT based on Loeffler's 8-point 1-D DCT algorithm
- Quantization of DCT coefficients and reordering them in ZigZag order
- Huffman encoding

The result is saved as a valid baseline JPEG file.

Inspired by: 
https://greasyfork.org/en/scripts/39428-bv7-jpeg-encoder-b/code

The [second script](dct_visualise.py) is an auxiliary utility designed to help understand how 
lossy still image compression algorithms based on block transforms (e.g., 
DCT or wavelet transforms) work. It processes an image by:
- Splitting it into 8x8 blocks
- Performing a 2D DCT transform
- Applying subquantization
- Filtering by zeroing components with magnitudes below a threshold
- Applying the inverse DCT to visualize the post-transform image quality

Finally, the result is saved into a ZIP file using Deflate compression 
(essentially LZ77 combined with Huffman coding), similar to many other 
image codecs, to demonstrate the effect of entropy encoding on images in 
different domains.

Note: The second script is NOT a valid JPEG encoder but provides a clear 
foundation for understanding the principles behind such compression 
algorithms.

## Usage
### [First Script](jpeg_coder.py)
The first script takes three command-line arguments:
- `[INPUT_FILENAME]`: Path to the input BMP file.
- `[OUTPUT_FILENAME]`: Name of the output JPEG file.
- `[QUALITY]`: An integer in the range from 0 (lowest quality) to 100 
(highest quality).

Example usage:
```sh
./jpeg_coder.py lenna.bmp lenna.jpg 90
```

### [Second Script](dct_visualise.py)
The second script takes one command-line argument:
- `[INPUT_FILENAME]`: Path to the input BMP file.

Example usage:
```sh
./dct_visualise.py lenna.bmp
```

### Get Started
To test the scripts, you can use the following command to download and 
convert an example image (requires ImageMagick 7 or later):
```sh
curl -sL 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_(test_image).png' | magick png:- bmp:- > lenna.bmp
```

## License
The **Software** in this project is licensed under the [MIT License](LICENSE).

