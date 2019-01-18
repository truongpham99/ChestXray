""" Utility functions 
"""
import cv2
import os
import glob
import numpy as np
from math import floor

class ImageClass():
    def __init__(self, image_paths, label):
        self.image_paths = image_paths
        self.label = label
    def __str__(self):
        return "label: " + self.label + ", number of samples: " + str(len(self))
    def __len__(self):
        return len(self.image_paths)
    
def get_data(path):
    data = []
    for label in get_next_path(path):
        data_path = os.path.join(path,label)
        if label == "PNEUMONIA":
            virus_img, bacteria_img = get_virus_and_bacteria(data_path)
            data.extend([ImageClass(virus_img, "virus"),ImageClass(bacteria_img, "bacteria")])
        else:
            image_paths = [img for img in glob.glob(data_path + "/*.jpeg")]
            data.append(ImageClass(image_paths, label.lower()))
    return data

def get_binary_data(path):
    data = []
    for label in get_next_path(path):
        data_path = os.path.join(path,label)
        image_paths = [img for img in glob.glob(data_path + "/*.jpeg")]
        data.append(ImageClass(image_paths, label.lower()))
    return data

def get_virus_and_bacteria(path):   
    virus_img = glob.glob(path + "/*virus*.jpeg")
    bacteria_img = glob.glob(path + "/*bacteria*.jpeg")
    return virus_img, bacteria_img

def get_next_path(path):
    nxt_path = os.listdir(path)
    for pth in nxt_path:
        full_pth = os.path.join(path, pth)
        if not os.path.isdir(full_pth):
            nxt_path.remove(pth)
    return nxt_path

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for subset in dataset:
        image_paths_flat += subset.image_paths
        labels_flat += [subset.label] * len(subset)
    return image_paths_flat, labels_flat

def load_images(image_paths, image_size):
    images = np.zeros((len(image_paths), image_size, image_size, 3))
    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i], cv2.COLORSPACE_GRAY)
        if img.ndim == 2:
            img = to_rgb(img)
        # might add preprocessing
        img = cv2.resize(img, (224,224))
        if img.shape != (224,224,3):
            print("Wrong dimension")
            return
        images[i,:,:,:] = img
    return images

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret 

# reserve image ratio

def resize(image):
    h,w = image.shape
    ratio = w/h
    if ratio == 1:
        image = cv2.resize(image, (224,224))
    elif ratio > 1:
        w_r = floor(224*ratio)
        image = cv2.resize(image, (w_r,224))

        # accounts for the black spaces
        w_box = floor((w_r - 224)/2)
        image = image[:,w_box:w_box+224]
    elif ratio < 1:
        h_r = floor(224*(1/ratio))
        image = cv2.resize(image, (224,h_r))

        # accounts for the liver
        image = image[:224,:]

    return image
