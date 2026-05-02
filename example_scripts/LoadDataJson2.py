# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:39:06 2023

@author: Aaron
"""
import json
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import os
from PIL import Image

with open('imageMap.json', 'r') as f:
    data = json.load(f)

path = os.getcwd()

keys = data.keys()

keys_list = list(keys)




entry = keys_list[5002]

img_file = entry.split('/')[1:]

file = path + os.sep + img_file[0] + os.sep + img_file[1] + '.png'


im = plt.imread(file)

im = Image.open(file)
source_image = Image.open(file)
palette = source_image.getpalette()
width, height = im.size
# Setting the points for cropped image
left = 0
top = 0
right = 70
bottom = 70
 
# Cropped image of above dimension
# (It will not change original image)
#the image's left upper point is (0, 0),
im1 = im.crop((left, top, right, bottom))

# plt.imshow()

# plt.imshow(im[:,:1000])
data[entry]
fps = data[entry]['fps']
numFrames = data[entry]['numFrames']
totalFrames = data[entry]['totalFrames']
frameWidth = data[entry]['frameWidth']
frameHeight = data[entry]['frameHeight']
numDirections = data[entry]['numDirections']

# numFrames = data[entry]['fps']
# numDirectionsFrames = data[entry]['numDirections']
# frameWidth = data[entry]['totalFrames']
# frameHeight = data[entry]['frameHeight']

numImgs = int(totalFrames/numDirections)
numSzenes = int(totalFrames/numImgs)

data[entry]['frameOffsets'][0]

# https://note.nkmk.me/en/python-pillow-paste/
# zeros = np.zeros((100,100,4))
# Image.fromarray(zeros, "P")
canvas = Image.new("P", (100, 100),(0,0,0,0))
canvas.putpalette(palette)






canvasWidth, canvasHeight = canvas.size
frame_arrays = []
for c in data[entry]['frameOffsets']:
    for i in c:
        print(i)
        # canvas = Image.new("P", (100, 100),(0,0,0,0))
        canvas = Image.new("P", (100, 100),(255,255,255,255))
        (255,255,255,255)
        canvas.putpalette(palette)
        x = i['x']
        y = i['y']
        w = i['w']
        h = i['h']
        sx = i['sx']
        ox = i['ox']
        oy = i['oy']
        # t = Image.fromarray(canvas, "P")
        
        left = canvasWidth//2 - ((w// 2)-ox)
        top = canvasHeight//4*3 - (h-oy)
        # left = x - (frameWidth// 2) + 100//2
        # top  = y - frameHeight + 100//2
        # im.crop((left, top, right, bottom))
        crop_image = source_image.crop((sx, 0, sx+frameWidth, frameHeight))
        canvas.paste(crop_image,(left,top))
        # rgb_image = canvas.convert('RGB', palette=Image.Palette.ADAPTIVE, colors=256)
        # rgb_image = Image.new('RGB', canvas.size, 'white')
        # rgb_image.paste(canvas, mask=canvas)
        rgb_image = canvas.convert('RGB')
        frame_array = np.array(rgb_image)
        black_pixels = (frame_array == [0, 0, 0]).all(axis=-1)
        frame_array[black_pixels] = [255, 255, 255]
        # crop_image =  im[:,sx:sx+frameWidth]
        # crop_image  = Image.fromarray(crop_image, "P")
        # crop_image.show()
        # solo_img = im[:,sx:sx+frameWidth]
        # canvas.paste(crop_image,(left,top))
        # np.array(canvas).shape
        plt.figure(figsize=(5,5))
        plt.imshow(frame_array)
        plt.show()
        frame_arrays.append(frame_array)
        # canvas.show()
        # solo_img = im[:,sx:sx+frameWidth]
        # cut_img = solo_img[:h,:w]
          # Define this before the loop
        # frame_arrays.append(frame_array)
        # plt.imshow(cut_img)
        # plt.show()

frame_arrays = np.array(frame_arrays)

frame_arrays.shape
# frame.shape
Arr = np.zeros((numDirections, numImgs, 100, 100,3 )).astype(np.uint8)
Arr
for i in np.arange(numImgs):
    for num, frame in enumerate(frame_arrays[i::numImgs]):
        Arr[num,i] = frame
        # plt.imshow(frame)
        # plt.show()


Arr.shape
Arr[3,5].shape
plt.imshow(Arr[3,1])
plt.show()

np.save(img_file[1], Arr)

