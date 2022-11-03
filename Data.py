import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F

from Augmentations import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, ToTensor1

class Sequntial_RGB_Dataset(Dataset):
    """
    A custom dataset for loading images as sequences.
    Loads mask of images and labels.

    Params:
        frames      : Dict of paths of images per sequence and corresponding labels. (Format from load images)
        n_classes   : Number of classes the dataset belongs to. so that the one hot embedding can be created 
        image_format: The image should be loaded in gray (96*96) or rgb (3*224*224) format and dimensions
        transform   : The transforms to be applied during training
        dataset     : Livecellminer or Zhong Morphology .. these two have different naming formats for mask images.

    Return:
        "image"  : images belong to the sequences (gray: [batch_size, seq_len, 1, 96, 96]) (RGB: [batch_size, seq_len, 3, 224, 224] )
        "mask"   : masks belong to the sequence with same dimensions as images. (These are masked images)
        "label"  : One hot encoded labels belong to same sequence [batch_size, seq_len]
        "folder" : name of the sequence folder
        "true"   : the values of original labels given with dataset [not one hot]
    """

    def __init__(self, frames, n_classes = 3, image_format = 'gray', transform = None, dataset = "LivecellMiner"):
        super().__init__()
        self.frames = frames
        self.transform =  transform
        self.image_format = image_format
        self.n_classes = n_classes
        self.dataset = dataset

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        #load the paths and labels from the self.frames dictionary
        sequence = self.frames[idx][0]
        labels = np.array(self.frames[idx][1])
        true_labels = np.array(self.frames[idx][2])

        #Blank list to append the images and masks per sequence 
        images = []
        masks  = []

        #read the name of the folder from the first image path.
        folder = sequence[0].split("/")[-3]

        #iterate through each imagepath in a sequence in proper sorted order
        for path in sequence:

            #load image path 
            img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
            #for rgb image change to the required dimensions
            if self.image_format == 'RGB':
                img = cv2.resize(img,(224,224))
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                img = np.moveaxis(img, -1, 0)
            else:
                img = img[None,:]

            #load mask path
            mask_path = path.replace("/Raw","/Mask")
            #naming of LivecellMiner cells are diffrernt
            if self.dataset == 'LivecellMiner':
                mask_path = path.replace("Raw","Mask")
            mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)
            #for rgb image change to the required dimensions
            if self.image_format == 'RGB':
                mask = cv2.resize(mask,(224,224))
                mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
                mask = np.moveaxis(mask, -1, 0)
            else:
                mask = mask[None,:]

            #append the images and masks belong to same sequence to the lists
            masks.append(mask)
            images.append(img)
        
        #normalize images  masks and change to tensor
        images = np.array(images, dtype= np.float32)                       
        images = images/ max(np.unique(images))
        images = torch.tensor(images, dtype=torch.float32)
        #normalize masks and change to tensor (as masks it is change to masked image)
        masks =np.array(masks, dtype= np.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        masks = torch.mul(masks,images)

        # apply the transforms to both images and masks
        if self.transform:
            images,masks = self.transform(images,masks)

        #create one hot encoding of the labels and assign to tensor
        labels = torch.tensor(labels)
        labels = F.one_hot(labels, num_classes=self.n_classes)
        labels = labels.type(torch.float32)

        #true labels are created without one hot encoding
        true_labels = torch.tensor(true_labels)

        sample = { "image" : images , "mask" : masks, "label" : labels, "folder" : folder, "true": true_labels}
        return sample



class ImageNet_Dataset(Dataset):
    """
    A custom dataset for loading images in RGB imagenet format. Here the images are loaded independent of sequence. 
    Loads mask of images and labels.

    Params:
        frames      : Dict of paths of images per sequence and corresponding labels. (Format from load images)
        n_classes   : Number of classes the dataset belongs to. so that the one hot embedding can be created 
        transform   : The transforms to be applied during training

    Return:
        "image"  : images belong to the sequences  (RGB: [batch_size, 3, 224, 224] )
        "mask"   : masks belong to the sequence with same dimensions as images. (These are masked images)
        "label"  : One hot encoded labels belong to same sequence [batch_size, seq_len]
        "folder" : name of the sequence folder
        "true"   : the values of original labels given with dataset [not one hot]
    """

    def __init__(self, frames, n_classes = 3, transform = None):
        super().__init__()
        self.frames = frames
        self.transform = transform
        self.n_classes = n_classes

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        #load the paths and labels from the self.frames dictionary
        imagepath = self.frames[idx][0]
        labels = np.array(self.frames[idx][1])
        
        #read the name of the folder from the first image path.
        folder = imagepath.split("/")[-2] + imagepath.split("/")[-1]

        #read the image and change it to rgb format
        img = cv2.imread(imagepath,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img,(224,224))
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = np.moveaxis(img, -1, 0)
        #read the corresponding mask and change it to rgb format
        mask_path = imagepath.replace("/Raw","/Mask")
        mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask,(224,224))
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        mask = np.moveaxis(mask, -1, 0)

        #normalize the image and change to tensor
        images = np.array(img, dtype= np.float32)                       
        images = images/ max(np.unique(images))
        images = torch.tensor(images, dtype=torch.float32)   
        #normalize the image and change to tensor. (mask to masked image)
        masks =np.array(mask, dtype= np.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        masks = torch.mul(masks,images)
        
        #apply augmentations
        if self.transform:
            images,masks = self.transform(images,masks)

        #true labels are created without one hot encoding
        true_labels = torch.tensor(labels)

        #create one hot encoding of labels
        labels = torch.tensor(labels)
        labels = F.one_hot(labels, num_classes=self.n_classes)
        labels = labels.type(torch.float32)

        sample = { "image" : images , "mask" : masks, "label" : labels, "folder" : folder, 'true': true_labels}
        return sample