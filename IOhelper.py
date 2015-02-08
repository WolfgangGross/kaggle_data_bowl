import glob
import os
import numpy as np
from pylab import cm
from scipy import ndimage
from skimage.feature import peak_local_max

def get_dir_names():
    # get the class names from the directory structure
    directory_names = list(set(glob.glob(os.path.join("train", "*"))).difference(set(glob.glob(os.path.join("train","*.*")))))
    return directory_names

# Rescale the images and create the combined metrics and training labels

def get_num_img(directory_names):
    #get the total training images
    numberofImages = 0
    for folder in directory_names:
        for fileNameDir in os.walk(folder):   
            for fileName in fileNameDir[2]:
                # Only read in the images
                if fileName[-4:] != ".jpg":
                    continue
                numberofImages += 1
    return numberofImages

def init_x_and_y(maxPixel=25, addi_features=1):
    # We'll rescale the images to be 25x25
    imageSize = maxPixel * maxPixel
    num_rows = get_num_img(get_dir_names()) # one row for each image in the training dataset
    num_features = imageSize + addi_features # for our rati; evtl. add more columns for additional ratios

    # X is the feature vector with one row of features per image
    # consisting of the pixel values and our metric
    X = np.zeros((num_rows, num_features), dtype=float)
    # y is the numeric class label 
    y = np.zeros(num_rows)
    return X,y


def get_class_names():
    # List of string of class names    
    namesClasses = list()
    for folder in get_dir_names():
        # Append the string class name for each class
        currentClass = folder.split(os.pathsep)[-1]
        namesClasses.append(currentClass)
    return namesClasses

