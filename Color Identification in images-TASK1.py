#!/usr/bin/env python
# coding: utf-8

# In[86]:


#Task1
#Color identification in images
#Adithya Vangaru-GRIPMAY'23

#Importing Libraries

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sn
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:


#Each pixel is represented as a combination of three colors RGB
image = cv2.imread('sample.jpg')
print("The type of this input is {}".format(type(image)))
print("Shape: {}".format(image.shape))
plt.imshow(image)

## Output
# The type of this input is <class 'numpy.ndarray'>
# Shape: (3456, 4608, 3)


# In[88]:


#moving from BGR color space to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[89]:


#output with the colormap as gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')


# In[90]:


#Resizing the image
resized_image = cv2.resize(image, (1200, 600))
plt.imshow(resized_image)


# In[91]:


#Color Identification
#RGB to Hex Conversion

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# In[92]:


#Reading image in RGB color space

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# In[93]:


#Obtaining colors from an image
modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)


# In[94]:


#using Numpy’s reshape function to reshape the image data.
clf = KMeans(n_clusters = 8, init = 'k-means++',
            max_iter = 300, n_init = 10, random_state = 0)
labels = clf.fit_predict(modified_image)


# In[95]:


#Using KMeans Algorithm, we group using clusters
def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    return rgb_colors


# In[96]:


#To avoid warnings
import warnings
warnings.filterwarnings("ignore")


# In[97]:


#pie chart appearing with top 8 colors of the image.
get_colors(get_image('sample.jpg'), 8, True)


# In[98]:


#Searching Images using color
IMAGE_DIRECTORY = (r"C:\Users\Yaminisurya\Downloads\images")
COLORS = {
    'GREEN': [0, 128, 0],
    'BLUE': [0, 0, 128],
    'YELLOW': [255, 255, 0]
}
images = []

for file in os.listdir(IMAGE_DIRECTORY):
    if not file.startswith('.'):
        images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))


# In[99]:


#Displaying all the images

plt.figure(figsize=(20, 10))
for i in range(len(images)):
    plt.subplot(1, len(images), i+1)
    plt.imshow(images[i])


# In[100]:


#Matching Images with color
def match_image_by_color(image, color, threshold = 60, number_of_colors = 10):
    
    image_colors = get_colors(image, number_of_colors, False)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            select_image = True
    
    return select_image


# In[101]:


#Showing selected images
def show_selected_images(images, color, threshold, colors_to_match):
    index = 1
    
    for i in range(len(images)):
        selected = match_image_by_color(images[i],
                                        color,
                                        threshold,
                                        colors_to_match)
        if (selected):
            plt.subplot(1, 5, index)
            plt.imshow(images[i])
            index += 1


# In[102]:


#Searching for green
# Variable 'selected_color' can be any of COLORS['GREEN'], COLORS['BLUE'] or COLORS['YELLOW']
plt.figure(figsize = (20, 10))
show_selected_images(images, COLORS['GREEN'], 60, 5)


# In[103]:


##Searching for blue
# Variable 'selected_color' can be any of COLORS['GREEN'], COLORS['BLUE'] or COLORS['YELLOW']
plt.figure(figsize = (20, 10))
show_selected_images(images, COLORS['BLUE'], 60, 5)


# In[104]:


##Searching for yellow
# Variable 'selected_color' can be any of COLORS['GREEN'], COLORS['BLUE'] or COLORS['YELLOW']
plt.figure(figsize = (20, 10))
show_selected_images(images,COLORS['YELLOW'], 60, 5)

