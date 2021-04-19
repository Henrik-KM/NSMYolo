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
from utils.utils import *
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto() #Use to fix OOM problems with unet
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
#unet = tf.keras.models.load_model('../../input/network-weights/unet-1-dec-1415.h5',compile=False)
print_labels = False



# Particle params
Int = lambda : 1e-3*(0.1+0.8*np.random.rand())#1e-3*(0.1+0.8*np.random.rand())#1e-4#
Ds = lambda: 0.10*(0.05 + 1*np.random.rand())#0.10*np.sqrt((0.05 + 1*np.random.rand()))#0.02#
st = lambda: 0.04 + 0.01*np.random.rand()

# Noise params
dX=.00001+.00003*np.random.rand()
dA=0
noise_lev=.0001
biglam=0.6+.4*np.random.rand()
bgnoiseCval=0.03+.02*np.random.rand()
bgnoise=.08+.04*np.random.rand()
bigx0=.1*np.random.randn()

# def generate_trajectories(image,Int,Ds,st,nump):
#     vel = 0
#     length=image.shape[1]
#     times=image.shape[0]
#     x=np.linspace(-1,1,length)
#     t=np.linspace(-1,1,times)
#     X, Y=np.meshgrid(t,x)
#     f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
    
#     for p_nbr in range(nump):
#         I = Int()
#         D = Ds()
#         s = st()
        
#         # Generate trajectory 
#         x0=0
#         x0+=np.cumsum(vel+D*np.random.randn(times))
#         v1=np.transpose(I*f2(1,x0,s,0,Y))
        
#         # Save trajectory with intensity in first image
#         image[...,0] *= (1-v1)##(1-v1)

#         # Add trajectory to full segmentation image image
#         particle_trajectory = np.transpose(f2(1,x0,0.05,0,Y))
#         image[...,1] += particle_trajectory 

#         # Save single trajectory as additional image
#         image[...,-p_nbr-1] = particle_trajectory  
        
#     return image

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
        x0=-1+2*np.random.rand()
        if np.random.rand()<1:
            unstuckLength = np.random.randint(times-times/16)
            x0+=np.cumsum(vel+D*np.random.randn(unstuckLength))               
            stuckLength = np.random.randint(times-unstuckLength)
            diffStuckConstant = (np.random.rand(stuckLength)-0.5)*0.002
            x0 = np.append(x0,x0[-1]+np.cumsum(diffStuckConstant))                      
            x0 = np.append(x0,x0[-1]+np.cumsum(+D*np.random.randn(times-(stuckLength+unstuckLength))))          
        else: 
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


nump = lambda:np.clip(np.random.randint(5),1,3)
times=8192
length=128
im = create_batch(1,times,length,nump)
plt.imshow(im[0,:,:,1].T,aspect='auto')

debug=False
#if train
# Each label has 5 components - image type,x1,x2,y1,y2
#Labels are ordered as follows: LabelID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM, where 
#X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
#Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
#WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
#HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT

trackMultiParticle = True
treshold=0.5
#try:          

    #%%
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
        
    if trackMultiParticle:
        YOLOLabels = YOLOLabelSingleParticleToMultiple(YOLOLabels[0],overlap_thres=0.7,xdim=length,ydim=times) #Higher threshold means more likely to group nearby particles
        if debug:
            plt.figure()
            ax = plt.gca()
            plt.imshow(im[0,:,:,0],aspect='auto')
            YOLOCoords = ConvertYOLOLabelsToCoord(YOLOLabels,xdim=length,ydim=times)
            for p,x1,y1,x2,y2 in YOLOCoords:
                if p ==0:
                    ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
                elif p == 1:
                    ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='orange'))
                elif p==2:
                    ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='black'))
            
#except:
#       print("Label generation failed. Continuing..")

#%%
#Test unet predictive power
plt.close('all')
nump = lambda: 3
#import tensorflow as tf
import tensorflow as tf
config = tf.compat.v1.ConfigProto() #Use to fix OOM problems with unet
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
for i in range(0,2):
    unet = tf.keras.models.load_model('../../input/network-weights/unet-14-dec-1700.h5',compile=False)
    
    times = 2048
    #im = create_batch(1,2048,512,nump)
    
    pred = unet.predict(np.expand_dims(im[...,0],axis=-1))
    plt.figure()
    plt.imshow(im[0,:,:,0],aspect='auto')
    plt.ylabel('Time (s)')
    xticks = plt.xticks()[0]*0.0295*4
    xticksStr = ['']*len(xticks)
    yticks = plt.yticks()[0]*0.00487
    yticksStr = ['']*len(yticks)
    for xtick in range(0,len(xticks)):
        xticksStr[xtick] = str(round(xticks[xtick],1)) 
    for ytick in range(0,len(yticks)):
        yticksStr[ytick] = str(round(yticks[ytick],1)) 
        
    plt.xticks(plt.xticks()[0],xticksStr)
    plt.xlim(0,127)
    plt.yticks(plt.yticks()[0],yticksStr)
    plt.ylim(0,2048)
    plt.title("Processed Kymograph",fontsize=26)
    plt.xlabel('Position (\u03BCm)',fontsize=26)
    plt.colorbar()
    plt.clim(-1, 1);
    plt.figure()
    plt.imshow(im[0,:,:,1],aspect='auto')
    plt.ylabel('Time (s)',fontsize=26)
    plt.title("True Trajectory",fontsize=26)
    plt.xlabel('Position (\u03BCm)',fontsize=26)
    plt.xticks(plt.xticks()[0],xticksStr)
    plt.xlim(0,127)
    plt.yticks(plt.yticks()[0],yticksStr)
    plt.ylim(0,2048)
    plt.colorbar()
    plt.clim(0, 1);
    pred = unet.predict(np.expand_dims(im[...,0],axis=-1))
    plt.figure()
    plt.imshow(pred[0,:,:,0],aspect='auto')
    plt.ylabel('Time (s)',fontsize=26)
    plt.title("Predicted Trajectory",fontsize=26)
    plt.xlabel('Position (\u03BCm)',fontsize=26)
    plt.xticks(plt.xticks()[0],xticksStr)
    plt.xlim(0,127)
    plt.yticks(plt.yticks()[0],yticksStr)
    plt.ylim(0,2048)
    plt.colorbar()
    plt.clim(0, 1);
    
    ConvertTrajToMultiBoundingBoxes(im,length=128,times=times,treshold=0.5,trackMultiParticle=True)
    
    # pred = unet.predict(np.expand_dims(im[...,0],axis=-1))
    # plt.figure()
    # plt.imshow(pred[0,:,:,0].T,aspect='auto')
    # plt.xlabel('t')
    # plt.title("Predicted Trajectory")
    # plt.xlabel('x')
    
#%%
import tensorflow as tf
from models import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
im = create_batch(1,128,128,lambda:2)
pred = np.expand_dims(im[:,:,:,1],-1)

plt.figure()
plt.imshow(pred[0,:,:,0],aspect='auto')
plt.xlabel('x')
plt.title("Processed Image")
plt.xlabel('t')
plt.figure()
plt.imshow(im[0,:,:,0],aspect='auto')
plt.xlabel('x')
plt.title("Processed Image")
plt.xlabel('t')

classes = load_classes("data/custom/classesNSM.names")
Tensor = torch.cuda.FloatTensor
predDS = skimage.measure.block_reduce(pred,(1,1,1,1)) 
predDS = Variable(torch.from_numpy(predDS).type(Tensor), requires_grad=False)
model = Darknet("config/yolov3-customNSM.cfg", img_size=128).to(device)
weights_path = "weights/yolov3_ckpt_28.pth"
model.load_state_dict(torch.load(weights_path))#,map_location=torch.device('cpu')))
model.eval()

predDS = torch.unsqueeze(predDS[...,0],1)
detections = model(torch.cat([predDS]*3,1))
detections = non_max_suppression(detections,conf_thres=0.8,nms_thres=0.4)
detections=detections[0]
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
if detections is not None:
     #detections = rescale_boxes(detections, 256, pred.shape[:2])
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
#%%
import tensorflow as tf
from models import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.autograd import Variable
dataPath = "C:/Users/ccx55/OneDrive/Documents/GitHub/NSMYOLO/data/Mixed data 17 jan bgstd (for diffusion)/"
files= os.listdir(dataPath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.close('all')
timesInitial = 128
times = timesInitial+ 128

for file in files[7:10]:
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
    #model = Darknet("config/yolov3-customNSM.cfg", img_size=128).to(device)
    #model = model.load_state_dict(torch.load("weights/yolov3_ckpt_28.pth"))
   # model.eval()
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

#%%
treshold=0.5

debug=True
trackMultiParticle = False
YOLOLabels = []
plt.close('all')

nump = im.shape[-1]-2
batchSize = im.shape[0]
#YOLOLabels =np.reshape([None]*batchSize*nump*5,(batchSize,nump,5)) #np.zeros((batchSize,nump,5))#np.reshape([None]*1*2*5,(1,2,5))#
#YOLOLabels = np.reshape([None]*5,(1,1,5))
for j in range(0,batchSize):
    for k in range(0,nump):
        particle_img = im[j,:,:,2+k]
        particleOccurence = np.where(particle_img>treshold)
        if np.sum(particleOccurence) <= 0:
            pass
            #YOLOLabels = np.delete(YOLOLabels,[j,k],1)
        else:
            trajTreshold = int(times/16)
            trajectoryOccurence = np.diff(particleOccurence[0])
            trajectories = particleOccurence[0][np.where(trajectoryOccurence>trajTreshold)]
            trajectories = np.append(0,trajectories)
            trajectories = np.append(trajectories,times)
            
            for traj in range(0,len(trajectories)-1): 
                particleOccurence = np.where(particle_img[trajectories[traj]:trajectories[traj+1],:]>treshold)
                constant = trajectories[traj]
                if traj != 0:
                    particleOccurence = np.where(particle_img[trajectories[traj]+trajTreshold:trajectories[traj+1],:]>treshold)
                    constant = trajectories[traj]+trajTreshold
            
                x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                y1,y2 = np.min(particleOccurence[0])+constant,np.max(particleOccurence[0])+constant
            
                try:
                    YOLOLabels =np.append(YOLOLabels,np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5)),1)
        
                except:
                    YOLOLabels = np.reshape([None]*5,(1,1,5))
                    YOLOLabels[0,0,:] = 0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)   
                
                if debug and traj == 0:
                    plt.figure()
                    ax = plt.gca()
                    plt.imshow(particle_img,aspect='auto')
    
                if debug:
                    import matplotlib.patches as pch                  
                    ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
                    #plt.imshow(particle_img,aspect='auto')
                    print(YOLOLabels)
                    print(str(x1)+"--"+str(x2)+"--"+str(y1)+"--"+str(y2))
    
    
        if trackMultiParticle:
            YOLOLabels = YOLOLabelSingleParticleToMultiple(YOLOLabels[0],overlap_thres=0.6,xdim=length,ydim=times) #Higher threshold means more likely to group nearby particles
            if debug:
                plt.figure()
                ax = plt.gca()
                plt.imshow(im[0,:,:,0],aspect='auto')
                YOLOCoords = ConvertYOLOLabelsToCoord(YOLOLabels,xdim=length,ydim=times)
                for p,x1,y1,x2,y2 in YOLOCoords:
                    p = int(p)
                    ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor=colors[p]))   
                    ax.text(x1,y1,(classes[p]),color = colors[p],fontsize=18)
#%%
plt.close('all')
for i in range(0,3):
    im = create_batch(1,8192,128*4,nump)
    im = skimage.measure.block_reduce(im,(1,64,1,1),np.mean)
    plt.figure()
    plt.imshow(im[0,:,:,1],aspect='auto')
    YOLOLabels = ConvertTrajToMultiBoundingBoxes(im,length=128,times=128,treshold=0.05,trackMultiParticle=True)
    
#%%

plt.close('all')
for i in range(0,2):
    im = create_batch(1,128,128*4,lambda:3)
    #im = skimage.measure.block_reduce(im,(1,64,1,1),np.mean)
    plt.figure()
    plt.imshow(im[0,:,:,1],aspect='auto')
    YOLOLabels = ConvertTrajToMultiBoundingBoxes(im,length=128,times=128,treshold=0.5,trackMultiParticle=True)
   
#%%
plt.close('all')
import matplotlib.patches as pch
for i in range(0,1):
    im = create_batch(1,8192,128*4,nump)
    im = skimage.measure.block_reduce(im,(1,64,1,1),np.mean)
    plt.figure()
    plt.imshow(im[0,:,:,1],aspect='auto')
    YOLOLabels = ConvertTrajToMultiBoundingBoxes(im,length=128,times=128,treshold=0.05,trackMultiParticle=True)
    
    flipim = np.flip(im,axis=(1))[0,:,:,:]
    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,128,128)
    temp = 127-flipLabels[:,4] 
    flipLabels[:,4] = 127-flipLabels[:,2] 
    flipLabels[:,2] = temp
    plt.figure()
    ax = plt.gca()
    plt.imshow(flipim[:,:,1],aspect='auto')
    for _,x1,y1,x2,y2 in flipLabels:
        ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
        
    flipim = np.flip(im,axis=(2))[0,:,:,:]
    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,128,128)
    temp = 127-flipLabels[:,3] 
    flipLabels[:,3] = 127-flipLabels[:,1] 
    flipLabels[:,1] = temp    
    plt.figure()
    ax = plt.gca()
    plt.imshow(flipim[:,:,1],aspect='auto')
    for _,x1,y1,x2,y2 in flipLabels:
        ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
        
    flipim = np.flip(im,axis=(1,2))[0,:,:,:]
    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,128,128)
    temp = 127-flipLabels[:,4] 
    flipLabels[:,4] = 127-flipLabels[:,2] 
    flipLabels[:,2] = temp    
    temp = 127-flipLabels[:,3] 
    flipLabels[:,3] = 127-flipLabels[:,1] 
    flipLabels[:,1] = temp    
    plt.figure()
    ax = plt.gca()
    plt.imshow(flipim[:,:,1],aspect='auto')
    for _,x1,y1,x2,y2 in flipLabels:
        ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
        
    flipim = np.rot90(im,axes=(1,2),k=1)[0,:,:,:]
    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,128,128)
    
    temp = np.copy(flipLabels[:,2]) 
    flipLabels[:,2] = 127-flipLabels[:,1]
    flipLabels[:,1] = temp
    
    temp = np.copy(flipLabels[:,4]) 
    flipLabels[:,4] = 127-flipLabels[:,3]
    flipLabels[:,3] = temp
    
    plt.figure()
    ax = plt.gca()
    plt.imshow(flipim[:,:,1],aspect='auto')
    temp = np.copy(flipLabels[:,2])
    flipLabels[:,2] = flipLabels[:,4]
    flipLabels[:,4] = temp
    #flipLabels[:,2], flipLabels[:,4] = flipLabels[:,4], flipLabels[:,2]
    for _,x1,y1,x2,y2 in flipLabels:
        ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))    
        
    flipim = np.rot90(im,axes=(1,2),k=3)[0,:,:,:]
    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,128,128)
    
    temp = np.copy(flipLabels[:,2]) 
    flipLabels[:,2] = flipLabels[:,1]
    flipLabels[:,1] = 127-temp
    
    temp = np.copy(flipLabels[:,4]) 
    flipLabels[:,4] = flipLabels[:,3]
    flipLabels[:,3] = 127-temp
    
    plt.figure()
    ax = plt.gca()
    plt.imshow(flipim[:,:,1],aspect='auto')
    temp = np.copy(flipLabels[:,1])
    flipLabels[:,1] = flipLabels[:,3]
    flipLabels[:,3] = temp
    for _,x1,y1,x2,y2 in flipLabels:
        ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))    


    # self.imSave[1,:,:,:] = np.flip(im,axis=(2))[0,:,:,:]
    # self.imSave[2,:,:,:] = np.flip(im,axis=(1,2))[0,:,:,:]
    
#%% Generate noise from exp. data
plt.close('all')
baseNoise1 = np.load('ferritin_highNoise.npy')
baseNoise2 = np.load('ferritin_lowNoise.npy')
baseNoise3 = np.load('ferritin_lowNoise_50_11.npy')

baseNoise = np.append(baseNoise1,baseNoise2,0)
baseNoise = np.append(baseNoise,baseNoise3,0)

im= create_batch(1,128,128*4,lambda:3)
im = np.expand_dims(im[:,:,:,1],-1)
startIndex = np.random.randint(np.size(baseNoise,0)-128)
plt.figure()
plt.imshow(baseNoise[startIndex:startIndex+128,:],aspect='auto')

im = im + np.expand_dims(baseNoise[startIndex:startIndex+128,:],(0,-1))
plt.figure()
plt.imshow(im[0,:,:,0],aspect='auto')

#%% Test stuck particles

im = create_batch(1,512,128,lambda:3)
#%%
plt.close('all')
plt.imshow(im[0,:,:,1],aspect='auto')
segment = np.copy(im[0,:,:,1])

ono = np.ones(im[0,:,:,1].shape)
conv_im = convolve2d(im[0,:,:,1],np.ones((1,20)),mode="same")

plt.figure()
plt.imshow(conv_im,aspect='auto')

#segment[segment<0.1] = 0

#diffTest = np.diff(segment,axis=0)
#diffTest[diffTest<0] = 0
# plt.figure()
# plt.imshow(diffTest,aspect='auto')
# plt.figure()
# diffTest2 = segment-diffTest
# plt.imshow(diffTest2,aspect='auto')
#%%
import skimage
from scipy.signal import convolve2d
import scipy
im = create_batch(1,512,128,lambda:1)
segment = np.copy(im[0,:,:,1])
plt.close('all')
M = 10
original_img = segment
for i in range(0,5):
    
    plt.imshow(original_img,aspect='auto')
    conv_img = original_img - convolve2d(original_img,np.ones((M,1))/M,mode='same')
    plt.figure()
    plt.imshow(conv_img,aspect='auto')
    
    img = original_img * (1-conv_img)
    img /= np.max(img)
    img[img<0.99] = 0
    img[img>0] = 1
    
    identifiedStuckTraj = np.sum(img,0)
    img[:,identifiedStuckTraj<2] = 0
    
    plt.figure()
    #plt.title(filename + ', j: ' + str(j)+ ' counter: ' + str(counter))
    plt.imshow(img,aspect='auto')
    plt.show()
    
    plt.figure(figsize=(14,6))
    binary_img = scipy.ndimage.morphology.binary_dilation(img,structure=np.ones((M,1)))
    plt.imshow(binary_img,aspect='auto')
    plt.show()
    
    idcs = np.sum(binary_img,axis=1)==0
    cut_img = original_img[idcs]
    plt.imshow(cut_img,aspect='auto')
    plt.title('Unstuck particle')
    plt.show()
    original_img = np.copy(cut_img)
#np.savetxt('unstuck_particle_segmentation.csv',cut_img,delimiter=',')

