from __future__ import division
# runfile('C:/Users/ccx55/OneDrive/Documents/GitHub/NSMYOLO/detect.py',args='--model_def config/yolov3-customNSM.cfg --weights_path weights/yolov3_ckpt_10.pth --class_path data/custom/classesNSM.names')
# 128Multi runfile('C:/Users/ccx55/OneDrive/Documents/GitHub/NSMYOLO/detect.py',args='--model_def config/yolov3-customNSMMulti.cfg --weights_path weights/yolov3_Multi_ckpt_3_128.pth --class_path data/custom/classesNSMMulti.names --img_size 128')
# 8192 runfile('C:/Users/ccx55/OneDrive/Documents/GitHub/NSMYOLO/detect.py',args='--model_def config/yolov3-customNSM.cfg --weights_path weights/yolov3_ckpt_Nopred_8192DSx128_kaggl3.pth --class_path data/custom/classesNSM.names --img_size 8192')
# 8192Multi runfile('C:/Users/ccx55/OneDrive/Documents/GitHub/NSMYOLO/detect.py',args='--model_def config/yolov3-customNSMMulti.cfg --weights_path weights/yolov3_Multi_ckpt_3_8192.pth --class_path data/custom/classesNSMMulti.names --img_size 8192')
from models import *
from utils.utils import *
from utils.datasetsNSMTest import *

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


train_unet = True
unet=None

if train_unet:
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto() #Use to fix OOM problems with unet
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    unet = tf.keras.models.load_model('../../input/network-weights/unet-14-dec-1700.h5',compile=False)

trackMultiParticle = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--overlap_thres", type=float, default=0.5, help="overlap thresshold for removing images with overlapping trajectories")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=8192, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    times = opt.img_size
    length = opt.img_size
    
    if opt.img_size==8192:
        model = Darknet(opt.model_def, img_size=128).to(device)
        times = 128
        length=128
    else: 
        model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.weights_path))
        else: 
            model.load_state_dict(torch.load(opt.weights_path,map_location=torch.device('cpu')))

    model.eval()  # Set in evaluation mode
    dataset = ListDataset("train",img_size=opt.img_size, augment=False, totalData = 6,unet = unet,trackMultiParticle=trackMultiParticle)#,normalized_labels=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
   # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        input_imgs = imgs
        # Configure input
        try:
            input_imgs = Variable(input_imgs.type(Tensor))
        except:
            input_imgs = torch.stack(input_imgs)
            input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs) #Predict bounding boxes
            #detections[0,:,4] = detections[0,:,4]*2
            #detections[0,:,5] = detections[0,:,5]*2
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres) #Remove overlapping bounding boxes - keep only the best one
            if detections[0] is not None:
                if len(detections[0]) > 1 and not trackMultiParticle and False:
                    #print("double bbox")
                    #prediction = detections
                    #break
                    detections = remove_traj_overlap(detections, opt.overlap_thres) #Remove images where trajectories are too close together 
                    #if detections[0][0] is None:
                    #    detections[0] = None
                    #break

    # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
               
        img = input_imgs[0,0,:,:] #Change first index here to allow different batch sizes, make for loop
       # plt.figure()
        fig, ax = plt.subplots(1)
        #plt.figure("realIm")
        #ax = plt.gca()
        ax.imshow(img.cpu(),aspect='auto')
        plt.title('Detected boxes')
        #ax.set_xlim(192,320)
        detections = detections[0]
    
         # Draw bounding boxes and labels of detections
        if detections is not None:
             # Rescale boxes to original image
             #detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
             #if opt.img_size==8192:
                 #detections = rescale_boxes_custom(detections, 128, [8192,128])
             #detections = rescale_boxes_custom(detections, 128, img.shape[:2])
             unique_labels = detections[:, -1].cpu().unique()
             n_cls_preds = len(unique_labels)
             bbox_colors = random.sample(colors, n_cls_preds)
             for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                 try:
                     print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
     
                     box_w = x2 - x1
                     box_h = y2 - y1
                         

                     color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                     # Create a Rectangle patch
                     bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                     ax.text(x1,y1,(classes[int(cls_pred)]),color = color,fontsize=18)
                     # Add the bbox to the plot
                     print(str(x1) + " " + str(y1) + " " + str(box_w) + " "+str(box_h))
                     ax.add_patch(bbox)
                     if opt.img_size ==8192:
                         locs, labels = plt.yticks()
                         locs = np.linspace(0,128,9)[0::2]
                         labels = np.linspace(0,8192,9)[0::2]
                         labels = [str(int(label)) for label in labels]
                         plt.yticks(locs,labels)
                 except: 
                         print("Flawed detection bbox")
        for _, x1, y1, x2, y2 in ConvertYOLOLabelsToCoord(targets[:,1:],times,length):
            box_w = x2 - x1
            box_h = y2 - y1
            ax = plt.gca()
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor='white', facecolor="none")
            ax.add_patch(bbox)
            
         # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        # plt.close()

#%%

#fig, ax = plt.subplots(1)
#ax.imshow(np.zeros((128,8192)),aspect='auto')
#rescale_boxes_custom(detections,128,[128,8192])

