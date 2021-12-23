#!/usr/bin/env python
# coding: utf-8

# ### __Import Libraries__

# In[1]:


import numpy as np
import pandas as pd
import os
import random
import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn import model_selection
#from sklearn.decomposition import PCA


# ### __Load Images__

# In[2]:


def get_images(dir_):
    images = []
    for subdir, dirs, files in os.walk(dir_):
        for file in files:
            if '.DS_Store' not in file:
                images.append(os.path.join(subdir, file))

    return list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), images))


# #### Non-vehicle images

# In[3]:


non_vehicles = get_images('non-vehicles/non-vehicles')
num_nonvehicles = len(non_vehicles)


# In[4]:


print(len(non_vehicles))
print(non_vehicles[0].shape)


# #### Vehicle images

# In[5]:


vehicles = get_images('vehicles/vehicles')
num_vehicles = len(vehicles)


# In[6]:


print(len(vehicles))
print(vehicles[0].shape)


# ### __Histogram of Gradients__

# In[7]:


def get_Hog_features(img, pix_per_cell = (10,10), cell_per_block = (2,2), visaulize=True):

    features = []

    if visaulize is True:
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=pix_per_cell,
                    cells_per_block=cell_per_block, visualize=visaulize,  multichannel =True)
        fd2, hog_img2 = hog(img, orientations=1, pixels_per_cell=pix_per_cell,
                    cells_per_block=cell_per_block, visualize=visaulize,  multichannel =True)
        features.append(fd2)
        features.append(fd)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)

        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        
        hog_image_rescaled2 = exposure.rescale_intensity(hog_img2, in_range=(0, 10))
        ax3.imshow(hog_image_rescaled2, cmap=plt.cm.gray)
        ax3.set_title('X-Gradients')
        
        plt.show()
        
    else:
        features.append(hog(img, orientations=1, pixels_per_cell=pix_per_cell,
                    cells_per_block=cell_per_block, visualize=visaulize,  multichannel =True))
        features.append(hog(img, orientations=8, pixels_per_cell=pix_per_cell,
                    cells_per_block=cell_per_block, visualize=visaulize,  multichannel =True))
        
    return np.concatenate(features)

def get_Hog_features(images,pix_per_cell=(10,10), cell_per_block = (2,2)):
    features = []

    for im in images:
        curr=[]
        curr.append(hog(im, orientations=1, pixels_per_cell=pix_per_cell,
                    cells_per_block=cell_per_block, visualize=False,  multichannel =True))
        curr.append(hog(im, orientations=8, pixels_per_cell=pix_per_cell,
                    cells_per_block=cell_per_block, visualize=False,  multichannel =True))
        features.append(np.concatenate(curr))
        
    return features
# #### An Example of Vehicle images

# In[8]:


features_v = get_Hog_features(vehicles[0])


# #### An Example of Non-vehicle images

# In[9]:


features_n = get_Hog_features(non_vehicles[0])


# In[37]:


print('number of features :',len(features_n))


# In[11]:


x = pd.DataFrame(columns=range(len(features_n)), dtype = 'float64')
for v in vehicles:
    x.loc[len(x)] = get_Hog_features(v, visaulize=False)
for n in non_vehicles:
    x.loc[len(x)] = get_Hog_features(n, visaulize=False)


# In[12]:


#pca = PCA(.99)
#pca.fit(features)
#features = pca.transform(features)


# In[15]:


y = pd.Series(np.concatenate((np.ones(num_vehicles), np.zeros(num_nonvehicles))))
print(np.ones(len(vehicles)))
print(np.zeros(len(non_vehicles)))
print(len(y))
x.head()


# ### __Split Training/Test__

# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2, train_size=0.8, random_state=42)
print(len(x_train), len(x_test))


# ### __Training the Classifier__

# In[35]:


svc = LinearSVC(C=0.1)
svc.fit(x_train, y_train)


# ### __Evaluate the Classifier__

# In[36]:


predicted = pd.Series(svc.predict(x_test), index=x_test.index)
indices = predicted[predicted ==1 ].index.tolist()

result = []
for i in indices:
    image = vehicles[0]
    if i < num_vehicles:
        image = vehicles[i]
    if i >= num_vehicles:
        image = non_vehicles[i - num_vehicles]
    
    # Threshold for color intensity in the image. But doesn't affect to result.
    #green (36,25,25), (86,255,255)
    #black (0,0,0), (255,255,0)
    #white (0,0,200), (180,255,255)
    #blue (100,150,0), (140,255,255)
    #yellow (20,100,100), (30,255,255)
    
    #green = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (36,25,25), (86,255,255))
    #g_ratio = len(image[green>0])/(len(image)*len(image))
    
    #if g_ratio > 0.7:
    #    print(g_ratio)
    #    result.append(i)
    #    plt.imshow(image, cmap=plt.cm.gray)
    #    plt.show()
        
#predicted.loc[result] = 0

cm = confusion_matrix(y_test, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
disp.plot()
plt.show()
tn, fp, fn, tp = cm.ravel()
print('true negative', tn)
print('false positive', fp)
print('false negative', fn)
print('true positive', tp)

print('False Positive Rate =', fp/(fp+tn))


# In[39]:


print('Accuracy :', round(accuracy_score(y_test, predicted), 4))


# ### __False Positive images__

# In[40]:


com = predicted.compare(y_test)
false_positive = com[com['self']==1].index.tolist()
print(false_positive)

for i in false_positive:
    print(i-num_vehicles)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    #plt.axis("off")
    plt.imshow(non_vehicles[i-num_vehicles])
    plt.show()


# In[ ]:




