#!/usr/bin/env python3
'''
===============================================================================
Filename: jpeg_coder.py
-------------------------------------------------------------------------------
Created on: 2025-06-24

License:
    MIT License
    Copyright (c) 2025 Kirill Kononenko
===============================================================================
'''
import struct
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import time


class JpegEncoder:

    def __init__(self, quality=50):
        self.zigZag = [
            0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12,
            17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32,
            39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50,
            56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
        ]

        # (ITU-T T.81¦ISO/IEC IS 10918-1)
        # K.3.3.1 Specification of typical tables for DC difference coding
        self.std_dc_luminance_nrcodes = [
            0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
        ]
        self.std_dc_luminance_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.std_dc_chrominance_nrcodes = [
            0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
        ]
        self.std_dc_chrominance_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # (ITU-T T.81¦ISO/IEC IS 10918-1)
        # K.3.3.2 Specification of typical tables for AC coefficient coding
        self.std_ac_luminance_nrcodes = [
            0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d
        ]
        self.std_ac_luminance_values = [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41,
            0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91,
            0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24,
            0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a,
            0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53,
            0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66,
            0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
            0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93,
            0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
            0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9,
            0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1,
            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
        ]

        self.std_ac_chrominance_nrcodes = [
            0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77
        ]
        self.std_ac_chrominance_values = [
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12,
            0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14,
            0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15,
            0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17,
            0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a,
            0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
            0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
            0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a,
            0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
            0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5,
            0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
            0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9,
            0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
        ]

        self.byteout = bytearray()
        self.bytenew = 0
        self.bytepos = 7

        self.quality = quality
        self.init_quant_tables()
        self.init_huffman_tables()
        self.init_category_and_bitcode()

        self.DCY = 0
        self.DCU = 0
        self.DCV = 0

    def init_quant_tables(self):
        if self.quality > 100:
            self.quality = 100

        self.quality = 100 - self.quality

        # (ITU-T T.81¦ISO/IEC IS 10918-1)
        # Table K.1 – Luminance quantization table
        # Also mentioned in:
        # Wallace, G. K. (1991). The JPEG still picture compression standard. Communications of the ACM, 34(4), 30–44.
        # doi:10.1145/103085.103089
        self.std_luminance_qt = np.array([
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14,13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
        ])

        # (ITU-T T.81¦ISO/IEC IS 10918-1)
        # Table K.2 – Chrominance quantization table
        self.std_chrominance_qt = np.array([
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99
        ])

        self.YTable = (self.std_luminance_qt * (self.quality/100)).astype(int) + 1
        self.UVTable = (self.std_luminance_qt * (self.quality/100)).astype(int) + 1

        self.fdtbl_Y = np.zeros(64)
        self.fdtbl_UV = np.zeros(64)

        self.fdtbl_Y = 1.0 / self.YTable
        self.fdtbl_UV = 1.0 / self.UVTable

    def build_huffman_table(self, nrcodes, values):
        code = 0
        pos = 0
        huffsize = []
        huffcode = []

        for i in range(1, 17):
            for _ in range(nrcodes[i - 1]):
                huffsize.append(i)
        huffsize.append(0)

        k = 0
        si = huffsize[0]
        while True:
            while huffsize[k] == si:
                huffcode.append(code)
                code += 1
                k += 1
            if huffsize[k] == 0:
                break
            while si < huffsize[k]:
                code <<= 1
                si += 1

        huffman_table = {}
        for i, val in enumerate(values):
            huffman_table[val] = (huffcode[i], huffsize[i])
        return huffman_table

    def init_huffman_tables(self):
        self.YDC_HT = self.build_huffman_table(self.std_dc_luminance_nrcodes,
                                               self.std_dc_luminance_values)
        self.UVDC_HT = self.build_huffman_table(
            self.std_dc_chrominance_nrcodes, self.std_dc_chrominance_values)
        self.YAC_HT = self.build_huffman_table(self.std_ac_luminance_nrcodes,
                                               self.std_ac_luminance_values)
        self.UVAC_HT = self.build_huffman_table(
            self.std_ac_chrominance_nrcodes, self.std_ac_chrominance_values)

    def init_category_and_bitcode(self):
        self.category = {}
        self.bitcode = {}

        for i in range(-32767, 32768):
            abs_i = abs(i)
            if abs_i == 0:
                self.category[i] = 0
                self.bitcode[i] = (0, 0)
            else:
                nbits = abs_i.bit_length()
                self.category[i] = nbits
                if i >= 0:
                    self.bitcode[i] = (i, nbits)
                else:
                    val = ((1 << nbits) - 1) + i
                    self.bitcode[i] = (val, nbits)

    def write_byte(self, val):
        self.byteout.append(val & 0xFF)

    def write_word(self, val):
        self.write_byte((val >> 8) & 0xFF)
        self.write_byte(val & 0xFF)

    def write_bits(self, val, length):
        for i in reversed(range(length)):
            bit = (val >> i) & 1
            if bit:
                self.bytenew |= (1 << self.bytepos)
            self.bytepos -= 1
            if self.bytepos < 0:
                self.write_byte(self.bytenew)
                if self.bytenew == 0xFF:
                    self.write_byte(0x00)
                self.bytenew = 0
                self.bytepos = 7

    # APP0: JPEG Application Segment 0 used in JFIF files to store metadata such as version, density, and thumbnail info.
    def write_APP0(self):
        self.write_word(0xFFE0)
        self.write_word(16)
        self.write_byte(ord('J'))
        self.write_byte(ord('F'))
        self.write_byte(ord('I'))
        self.write_byte(ord('F'))
        self.write_byte(0)
        self.write_byte(1)
        self.write_byte(1)
        self.write_byte(0)
        self.write_word(1)
        self.write_word(1)
        self.write_byte(0)
        self.write_byte(0)

    # Define Quantization Table(s)
    # For information on marker codes, see:
    # (ITU-T T.81¦ISO/IEC IS 10918-1)
    # Table B.1 – Marker code assignments
    def write_DQT(self):
        self.write_word(0xFFDB)
        self.write_word(132)
        self.write_byte(0)
        for i in range(64):
            self.write_byte(self.YTable[i])
        self.write_byte(1)
        for i in range(64):
            self.write_byte(self.UVTable[i])

    # Start Of Frame (Baseline DCT)
    # Indicates that this is a baseline DCT-based JPEG, and specifies the width, height, number of components, and component subsampling
    # For information on marker codes, see:
    # (ITU-T T.81¦ISO/IEC IS 10918-1)
    # Table B.1 – Marker code assignments
    def write_SOF0(self, width, height):
        self.write_word(0xFFC0)
        self.write_word(17)
        self.write_byte(8)
        self.write_word(height)
        self.write_word(width)
        self.write_byte(3)
        self.write_byte(1)
        self.write_byte(0x11)
        self.write_byte(0)
        self.write_byte(2)
        self.write_byte(0x11)
        self.write_byte(1)
        self.write_byte(3)
        self.write_byte(0x11)
        self.write_byte(1)

    # For information on marker codes, see:
    # (ITU-T T.81¦ISO/IEC IS 10918-1)
    # Table B.1 – Marker code assignments
    def write_DHT(self):

        def write_table(nrcodes, values, table_class, table_id):
            length = 2 + 1 + 16 + len(values)
            self.write_word(0xFFC4)
            self.write_word(length)
            self.write_byte((table_class << 4) | table_id)
            for i in range(16):
                self.write_byte(nrcodes[i])
            for v in values:
                self.write_byte(v)

        write_table(self.std_dc_luminance_nrcodes,
                    self.std_dc_luminance_values, 0, 0)
        write_table(self.std_ac_luminance_nrcodes,
                    self.std_ac_luminance_values, 1, 0)
        write_table(self.std_dc_chrominance_nrcodes,
                    self.std_dc_chrominance_values, 0, 1)
        write_table(self.std_ac_chrominance_nrcodes,
                    self.std_ac_chrominance_values, 1, 1)

    # Start Of Scan
    # Begins a top-to-bottom scan of the image.
    # This marker specifies which slice of data it will contain, and is immediately followed by entropy-coded data.
    # For information on marker codes, see:
    # (ITU-T T.81¦ISO/IEC IS 10918-1)
    # Table B.1 – Marker code assignments
    def write_SOS(self):
        self.write_word(0xFFDA)
        self.write_word(12)
        self.write_byte(3)
        self.write_byte(1)
        self.write_byte(0x00)
        self.write_byte(2)
        self.write_byte(0x11)
        self.write_byte(3)
        self.write_byte(0x11)
        self.write_byte(0)
        self.write_byte(0x3F)
        self.write_byte(0)

    # Fast DCT transform and quantization
    def fDCT_quant(self, data, fdtbl):
        data = data - 128
        block = data.reshape((8, 8)).astype(float)
        # Loeffler, C., Ligtenberg, A., & Moschytz, G. S. (n.d.). Practical fast 1-D DCT algorithms with 11 multiplications.
        # International Conference on Acoustics, Speech, and Signal Processing
        # doi:10.1109/icassp.1989.266596
        #===============================================================================
        for _ in range(2):
            block = block.T
            for i in range(8):
                s = block[i,:]
                # Stage 1
                tmp0 = s[0] + s[7]
                tmp1 = s[1] + s[6]
                tmp2 = s[2] + s[5]
                tmp3 = s[3] + s[4]
                tmp4 = s[3] - s[4]
                tmp5 = s[2] - s[5]
                tmp6 = s[1] - s[6]
                tmp7 = s[0] - s[7]

                # Stage 2
                tmp10 = tmp0 + tmp3
                tmp13 = tmp0 - tmp3
                tmp11 = tmp1 + tmp2
                tmp12 = tmp1 - tmp2

                tmp4_7 = np.zeros(2)
                tmp4_7[0] = tmp4*np.cos((3*np.pi)/16) + tmp7*np.sin((3*np.pi)/16)
                tmp4_7[1] = -tmp4*np.sin((3*np.pi)/16) + tmp7*np.cos((3*np.pi)/16)

                tmp5_6 = np.zeros(2)
                tmp5_6[0] = tmp5*np.cos(np.pi/16) + tmp6*np.sin(np.pi/16)
                tmp5_6[1] = -tmp5*np.sin(np.pi/16) + tmp6*np.cos(np.pi/16)

                # Stage 3
                block[i,0] = tmp10 + tmp11
                block[i,4] = tmp10 - tmp11

                tmp12_13 = np.zeros(2)
                tmp12_13[0] = tmp12*np.sqrt(2)*np.cos(np.pi/16) + tmp13*np.sqrt(2)*np.sin(np.pi/16)
                tmp12_13[1] = -tmp12*np.sqrt(2)*np.sin(np.pi/16) + tmp13*np.sqrt(2)*np.cos(np.pi/16)
                block[i,2] = tmp12_13[0]
                block[i,6] = tmp12_13[1]

                tmp14 = tmp4_7[1] + tmp5_6[0]
                tmp15 = tmp4_7[0] + tmp5_6[1]
                block[i,3] = tmp4_7[1] - tmp5_6[0]
                block[i,5] = tmp4_7[0] - tmp5_6[1]

                # Stage 4
                block[i,1] = tmp14 + tmp15
                block[i,7] = tmp14 - tmp15
                block[i,3] *= np.sqrt(2)
                block[i,5] *= np.sqrt(2)

                # Scale factor to get the same output as the DCT in SciPy with norm='ortho'
                block[i,:] /= (2*np.sqrt(2))
        #===============================================================================
        # The above block can be replaced with the following lines
        # (requires scipy as an additional dependency):
        #
        # from scipy.fftpack import dct
        # block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        quantized = np.round(block.flatten() * fdtbl).astype(int)
        return quantized

    def processDU(self, CDU, fdtbl, DC, HTDC, HTAC):
        DU_DCT = self.fDCT_quant(CDU, fdtbl)
        DU = np.zeros(64, dtype=int)
        for i in range(64):
            DU[self.zigZag[i]] = DU_DCT[i]

        Diff = DU[0] - DC
        DC = DU[0]

        if Diff == 0:
            code, length = HTDC[0]
            self.write_bits(code, length)
        else:
            cat = self.category[Diff]
            code, length = HTDC[cat]
            self.write_bits(code, length)
            bits, blen = self.bitcode[Diff]
            self.write_bits(bits, blen)

        end0pos = 63
        while end0pos > 0 and DU[end0pos] == 0:
            end0pos -= 1

        if end0pos == 0:
            code, length = HTAC[0x00]
            self.write_bits(code, length)
        else:
            i = 1
            while i <= end0pos:
                startpos = i
                while i <= end0pos and DU[i] == 0:
                    i += 1
                nrzeroes = i - startpos
                while nrzeroes > 15:
                    code, length = HTAC[0xF0]
                    self.write_bits(code, length)
                    nrzeroes -= 16
                val = DU[i] if i <= end0pos else 0
                cat = self.category[val]
                rs = (nrzeroes << 4) + cat
                code, length = HTAC[rs]
                self.write_bits(code, length)
                bits, blen = self.bitcode[val]
                self.write_bits(bits, blen)
                i += 1
            if end0pos != 63:
                code, length = HTAC[0x00]
                self.write_bits(code, length)

        return DC

    def rgb_to_yuv(self, r, g, b):
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
        v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
        return int(y), int(u), int(v)

    def encode(self, bmp_path, jpeg_path):
        with open(bmp_path, 'rb') as f:
            bmp = f.read()

        if bmp[0:2] != b'BM':
            raise ValueError('Not a BMP file')

        offset = struct.unpack('<I', bmp[10:14])[0]
        width = struct.unpack('<I', bmp[18:22])[0]
        height = struct.unpack('<I', bmp[22:26])[0]
        bpp = struct.unpack('<H', bmp[28:30])[0]

        if bpp != 24:
            raise ValueError('Only 24bpp BMP supported')

        row_padded = (width * 3 + 3) & (~3)
        pixels = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            row_start = offset + (height - 1 - y) * row_padded
            for x in range(width):
                i = row_start + x * 3
                b, g, r = bmp[i], bmp[i + 1], bmp[i + 2]
                pixels[y, x] = [r, g, b]

        Y = np.zeros((height, width), dtype=np.int16)
        U = np.zeros((height, width), dtype=np.int16)
        V = np.zeros((height, width), dtype=np.int16)

        for i in range(height):
            for j in range(width):
                r, g, b = pixels[i, j]
                y, u, v = self.rgb_to_yuv(r, g, b)
                Y[i, j] = y
                U[i, j] = u
                V[i, j] = v

        self.byteout = bytearray()
        self.bytenew = 0
        self.bytepos = 7
        self.DCY = 0
        self.DCU = 0
        self.DCV = 0

        self.write_word(0xFFD8)  # SOI
        self.write_APP0()
        self.write_DQT()
        self.write_SOF0(width, height)
        self.write_DHT()
        self.write_SOS()

        for y in range(0, height, 8):
            for x in range(0, width, 8):
                YDU = np.zeros(64, dtype=np.int16)
                UDU = np.zeros(64, dtype=np.int16)
                VDU = np.zeros(64, dtype=np.int16)
                for row in range(8):
                    for col in range(8):
                        yy = y + row
                        xx = x + col
                        pos = row * 8 + col
                        if yy < height and xx < width:
                            YDU[pos] = Y[yy, xx]
                            UDU[pos] = U[yy, xx]
                            VDU[pos] = V[yy, xx]
                        else:
                            YDU[pos] = 0
                            UDU[pos] = 0
                            VDU[pos] = 0
                self.DCY = self.processDU(YDU, self.fdtbl_Y, self.DCY,
                                          self.YDC_HT, self.YAC_HT)
                self.DCU = self.processDU(UDU, self.fdtbl_UV, self.DCU,
                                          self.UVDC_HT, self.UVAC_HT)
                self.DCV = self.processDU(VDU, self.fdtbl_UV, self.DCV,
                                          self.UVDC_HT, self.UVAC_HT)

        if self.bytepos >= 0:
            self.write_bits((1 << (self.bytepos + 1)) - 1, self.bytepos)
        self.write_word(0xFFD9)  # EOI

        with open(jpeg_path, 'wb') as f:
            f.write(self.byteout)

        print(f'JPEG saved to {jpeg_path}')


if __name__ == '__main__':
    encoder = JpegEncoder(quality=int(sys.argv[3]))
    encoder.encode(sys.argv[1], sys.argv[2])
