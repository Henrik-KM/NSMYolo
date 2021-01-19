from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal import convolve2d
import skimage.measure
import pandas as pd

import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto() #Use to fix OOM problems with unet
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
#unet = tf.keras.models.load_model('../../input/network-weights/unet-1-dec-1415.h5',compile=False)
print_labels = False

def ConvertTrajToBoundingBoxes(im,length=128,times=128,treshold=0.5):
    debug=False
    # Each label has 5 components - image type,x1,x2,y1,y2
    #Labels are ordered as follows: LabelID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM, where 
    #X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
    #Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
    #WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
    #HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
    

    
    try:            
            nump = im.shape[-1]-2
            batchSize = im.shape[0]
            YOLOLabels = np.zeros((batchSize,nump,5))#np.reshape([None]*1*2*5,(1,2,5))#
            for j in range(0,batchSize):
                for k in range(0,nump):
                    particle_img = im[j,:,:,2+k]
                    particleOccurence = np.where(particle_img>treshold)
                    if np.sum(particleOccurence) <= 0:
                        YOLOLabels = np.delete(YOLOLabels,[j,k],1)
                    else:
                        x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                        y1,y2 = np.min(particleOccurence[0]),np.max(particleOccurence[0])  

                        YOLOLabels[j,k,:] = 0, np.abs(x2+x1)/2/(length-1), (y2+y1)/2/(times-1),(x2-x1)/(length-1),(y2-y1)/(times-1)         

                        if debug:
                            import matplotlib.patches as pch
                            max_nbr_particles = 5
                            nbr_particles = max_nbr_particles
                            plt.figure()#,figsize=(10,2))
                            ax = plt.gca()
                            plt.imshow(particle_img,aspect='auto')
                            ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
                            plt.imshow(particle_img,aspect='auto')
                            print(YOLOLabels)
                            plt.colorbar()
                            print(str(x1)+"--"+str(x2)+"--"+str(y1)+"--"+str(y2))

    except:
           print("Label generation failed. Continuing..")

    

    return YOLOLabels

nump = lambda: np.clip(np.random.randint(5),1,3)


# Particle params
Int = lambda : 1e-3*(0.1+0.8*np.random.rand())
Ds = lambda: 0.10*np.sqrt((0.05 + 1*np.random.rand()))
st = lambda: 0.04 + 0.01*np.random.rand()

# Noise params
dX=.00001+.00003*np.random.rand()
dA=0
noise_lev=.0001
biglam=0.6+.4*np.random.rand()
bgnoiseCval=0.03+.02*np.random.rand()
bgnoise=.08+.04*np.random.rand()
bigx0=.1*np.random.randn()

def generate_trajectories(image,Int,Ds,st,nump):
    vel = 0
    length=image.shape[1]
    times=image.shape[0]
    x=np.linspace(-1,1,length)
    t=np.linspace(-1,1,times)
    X, Y=np.meshgrid(t,x)
    f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
    
    for p_nbr in range(nump):
        I = Int()
        D = Ds()
        s = st()
        
        # Generate trajectory 
        x0=0
        x0+=np.cumsum(vel+D*np.random.randn(times))
        v1=np.transpose(I*f2(1,x0,s,0,Y))
        
        # Save trajectory with intensity in first image
        image[...,0] *= (1-v1)##(1-v1)

        # Add trajectory to full segmentation image image
        particle_trajectory = np.transpose(f2(1,x0,0.05,0,Y))
        image[...,1] += particle_trajectory 

        # Save single trajectory as additional image
        image[...,-p_nbr-1] = particle_trajectory  
        
    return image

def gen_noise(image,dX,dA,noise_lev,biglam,bgnoiseCval,bgnoise,bigx0):
    length=image.shape[1]
    times=image.shape[0]
    x=np.linspace(-1,1,length)
    t=np.linspace(-1,1,times)
    X, Y=np.meshgrid(t,x)
    f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
    bgnoise*=np.random.randn(length)

    tempcorr=3*np.random.rand()
    dAmp=dA#*np.random.rand()
    shiftval=dX*np.random.randn()
    dx=0
    dx2=0
    dAmp0=0
    
    bg0=f2(1,bigx0,biglam,0,x)
    ll=(np.pi-.05)
    
    noise_img = np.zeros_like(image)
    for j in range(times):
        dx=(.7*np.random.randn()+np.sin(ll*j))*dX

        bgnoiseC=f2(1,0,bgnoiseCval,dx,x)
        bgnoiseC/=np.sum(bgnoiseC)
        bg=f2(1,bigx0+dx,biglam,0,x)*(1+convolve(bgnoise,bgnoiseC,mode="same"))
        dAmp0=dA*np.random.randn()
        bg*=(1+dAmp0)
        noise_img[j,:,0]=bg*(1+noise_lev*np.random.randn(length))+.4*noise_lev*np.random.randn(length)
    return noise_img, bg0

def post_process(image,bg0):             
    image[:,:,0]/=bg0 # Normalize image by the bare signal

    image[:,:,0]/=np.mean(image[...,0],axis=0)        
    image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0) # Subtract mean over image

    # Perform same preprocessing as done on experimental images
    ono=np.ones((200,1))
    ono=ono/np.sum(ono)
    image[:,:,0]-=convolve2d(image[:,:,0],ono,mode="same")
    image[:,:,0]-=convolve2d(image[:,:,0],np.transpose(ono),mode="same")

    image[:,:,0]-=np.expand_dims(np.mean(image[:,:,0],axis=0),axis=0)
    image[:,:,0]*=1000
    
    return image
        
def create_batch(batchsize,times,length,nump):
    nump = nump() # resolve nump for each batch
    batch = np.zeros((batchsize,times,length,nump+2))
    
    for b in range(batchsize):
        image = np.zeros((times,length,nump+2))
        
        # Add noise to image
        noise_image, bg0 = gen_noise(image,dX,dA,noise_lev,biglam,bgnoiseCval,bgnoise,bigx0)
        image = generate_trajectories(noise_image,Int,Ds,st,nump)
        
        # Post process
        image = post_process(image,bg0)
        
        batch[b,...] = image
    
    return batch




def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=128, augment=False, multiscale=False, normalized_labels=True,totalData=10,unet=None):
        self.img_files = ""

        self.label_files = ""
        self.img_size = img_size
        self.max_objects = 5
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.totalData = totalData
        self.unet = unet

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        print_labels = False
            
        length = 128
        times = 128
        
        batchsize = 1 

        im = create_batch(batchsize,times,length,nump)
        
        v1 = self.unet.predict(np.expand_dims(im[...,0],axis=-1))
        #plt.imshow(v1[0,:,:,0],aspect='auto')
        YOLOLabels = ConvertTrajToBoundingBoxes(im,length=length,times=times,treshold=0.5)
        
        # For training on iOC = 5e-4, D = [10,20,50] mu m^2/s
        # Range on Ds: 0.03 -> 0.08
        # Range on Is: 5e-3 = good contrast
        
        # Plot predictions of validation samples
        #YOLOLabels=ConvertTrajToBoundingBoxes(v1,batchSize,nump,length=length,times=times,treshold=0.5)
       # v1 = np.sum(v1[0,...],0).T #Place all particles in the same image
        # v1 = np.sum(v1,1).T
        # Extract image as PyTorch tensor
        v1 = np.squeeze(v1,0)
        img = transforms.ToTensor()(v1)
       #img = torch.from_numpy(v1)
        img = torch.cat([img]*3)
        
        #img = torch.cat([img]*3) # Convert to 3-channel image to simulate RGB information
        #print(img.shape)
        # Handle images with less than three channels ## defunct? 
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        targets = None       
        boxes = torch.from_numpy(YOLOLabels).reshape(-1,5)#torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        
        #print(img.shape)
       # plt.figure()
        #plt.imshow(img[0,:,:],aspect='auto')
        #print(targets)
        return "", img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i        
        targets = torch.cat(targets, 0)
        #targets[:,0] = 0 #replace sample index with 0
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return self.totalData