from __future__ import division
from models import *
from utils.utils import *
#from utils.datasetsNSMTest import *

import os
import sys
import time
import datetime
import argparse
#import numpy as np 

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import numpy as np

import tensorflow as tf
config = tf.compat.v1.ConfigProto() #Use to fix OOM problems with unet
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
unet = tf.keras.models.load_model('../../input/network-weights/unet-1-dec-1415.h5',compile=False)

trackMultiParticle = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)
model_def="config/yolov3-customNSM.cfg"
weights_path="weights/yolov3_ckpt_28.pth"
img_size = 128
# Set up model
model = Darknet(model_def, img_size=img_size).to(device)

if weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(weights_path)
else:
    # Load checkpoint weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path))
    else: 
        model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))

model.eval()  # Set in evaluation mode
#%%
dataPath = "C:/Users/ccx55/OneDrive/Documents/GitHub/NSMYOLO/data/Mixed data 17 jan bgstd (for diffusion)/"
files= os.listdir(dataPath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.close('all')
timesInitial = 128
times = timesInitial+ 128

for file in files[10:11]:
    file = np.load(dataPath+file)

    unet = tf.keras.models.load_model('../../input/network-weights/unet-1-dec-1415.h5',compile=False)
    pred = unet.predict(np.expand_dims(file[timesInitial:times,11:139],axis=(-1,0)))
    plt.figure()
    plt.imshow(pred[0,:,:,0],aspect='auto')
    plt.xlabel('x')
    plt.title("Processed Image")
    plt.xlabel('t')
    unet = tf.keras.models.load_model('../../input/network-weights/unet-14-dec-1700.h5',compile=False)
    pred = unet.predict(np.expand_dims(file[timesInitial:times,11:139],axis=(-1,0)))
    fig,ax=plt.subplots(1)
    plt.imshow(pred[0,:,:,0],aspect='auto')
    plt.xlabel('x')
    plt.title("Processed Image")
    plt.xlabel('t')
    plt.figure()
    plt.imshow(file[timesInitial:times,11:139],aspect='auto')
    plt.xlabel('x')
    plt.title("Processed Image")
    plt.xlabel('t')
    
    classes = load_classes("data/custom/classesNSM.names")
    Tensor = torch.cuda.FloatTensor
    pred = Variable(torch.from_numpy(pred).type(Tensor), requires_grad=False)
    pred = torch.unsqueeze(pred[...,0],1)
    detections = model(torch.cat([pred]*3,1))
    detections = non_max_suppression(detections)
    detections=detections[0]
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    if detections is not None:
         #detections = rescale_boxes(detections, 128, pred.shape[:2])
         unique_labels = detections[:, -1].cpu().unique()
         n_cls_preds = len(unique_labels)
         bbox_colors = random.sample(colors, n_cls_preds)
         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1                    

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, zorder=2,edgecolor="black", facecolor="none")
            plt.text(x1,y1,(classes[int(cls_pred)]),color = color)
            # Add the bbox to the plot
            print(str(x1) + " " + str(y1) + " " + str(box_w) + " "+str(box_h))
            ax.add_patch(bbox)

    
