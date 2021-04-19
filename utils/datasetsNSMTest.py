from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal import convolve2d
import skimage.measure
from utils.utils import *
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

import glob
import random

from PIL import Image
import torch
import torch.nn.functional as F


from torch.utils.data import Dataset
import torchvision.transforms as transforms
print_labels = False
runOnGPU = False
generateNoise = True

L_reduction_factor = 4
T_reduction_factor = 1


#This function takes a kymograph as input and outputs YOLO coordinates
def ConvertTrajToMultiBoundingBoxes(im,length=128,times=128,treshold=0.5,trackMultiParticle=False):
    # Each label has 5 components - image type,x1,x2,y1,y2
    #Labels are ordered as follows: LabelID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM, where 
    #X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
    #Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
    #WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
    #HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
    debug =False
    
    nump = im.shape[-1]-2
    batchSize = im.shape[0]
    YOLOLabels =np.reshape([None]*batchSize*1*5,(batchSize,1,5))
    for j in range(0,batchSize):
        if debug:
            fig,ax2= plt.subplots(1)
            plt.imshow(im[j,:,:,1],aspect='auto')
            plt.title('All bounding boxes - uncombined',fontsize=26)
            plt.xlabel('Position (\u03BCm)',fontsize=26)
            plt.ylabel('Time (s)',fontsize=26)
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
            
            plt.colorbar()
            plt.clim(0, 1);
            
        #Calculate YOLO boxes by tresholding perfect segmentations. Change this to directly using generated particle trajecotiry positions instead
        for k in range(0,nump):
            particle_img = im[j,:,:,2+k]

            particleOccurence = np.where(particle_img>treshold)
            if np.sum(particleOccurence) <= 0:
                continue
            else:
                trajTreshold = int(times/16)
                trajectoryOccurence = np.diff(particleOccurence[0])
                trajectories = particleOccurence[0][np.where(trajectoryOccurence>trajTreshold)]
                trajectories = np.append(0,trajectories)
                trajectories = np.append(trajectories,times)
                
                for traj in range(0,len(trajectories)-1): 
                    if debug and traj == 0:
                        plt.figure()
                        ax = plt.gca()
                        plt.imshow(particle_img,aspect='auto')
                        plt.title('Single Particle bounding boxes')
                        plt.xlabel('Position (\u03BCm)',fontsize=26)
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
                        
                        plt.colorbar()
                        plt.clim(0, 1);
                        
                    particleOccurence = np.where(particle_img[trajectories[traj]:trajectories[traj+1],:]>treshold)
                    if np.sum(particleOccurence[1]) <=0 or np.sum(particleOccurence[0]) <=0:
                        continue
                    constant = trajectories[traj]
                    if traj != 0:
                        particleOccurence = np.where(particle_img[trajectories[traj]+trajTreshold:trajectories[traj+1],:]>treshold)
                        constant = trajectories[traj]+trajTreshold
                
                    x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                    y1,y2 = np.min(particleOccurence[0])+constant,np.max(particleOccurence[0])+constant
                    
                    A = (x2-x1)*(y2-y1)
                    if A > 100:

                
                        if YOLOLabels[0,0,0] is None:
                            YOLOLabels = np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5))   
                        else:
                            YOLOLabels =np.append(YOLOLabels,np.reshape([0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)],(1,1,5)),1)
      
            
                        if debug:
                            import matplotlib.patches as pch                  
                            ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
                            ax2.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
                            print(YOLOLabels)
                            print(str(x1)+"--"+str(x2)+"--"+str(y1)+"--"+str(y2))
        
    #If tracking trajectories containing multiple particles, convert boxes to multi-particle boxes
    if trackMultiParticle and not np.isnan(np.array(YOLOLabels,dtype=float)).any():
        YOLOLabels = YOLOLabelSingleParticleToMultiple(YOLOLabels[0],overlap_thres=0.6,xdim=times,ydim=length) #Higher threshold means more likely to group nearby particles
        if debug:
            print("Converting to Multi-Particle Boxes")
            print(YOLOLabels)
            plt.figure()
            ax = plt.gca()
            plt.imshow(im[0,:,:,0],aspect='auto')
            plt.title('All bounding boxes - combined',fontsize=26)
            plt.xlabel('Position (\u03BCm)',fontsize=26)
            plt.ylabel('Time (s)',fontsize=26)
            
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
            plt.colorbar()
            plt.clim(-1, 1);
            YOLOCoords = ConvertYOLOLabelsToCoord(YOLOLabels,xdim=times,ydim=length)
            classes = ['particle','two particles','three+ particles']
            colors = ['white','orange','black']
            for p,x1,y1,x2,y2 in YOLOCoords:
                p = int(p)
                ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor=colors[p]))   
                ax.text(x1,y1,(classes[p]),color = colors[p],fontsize=18)
                    
    return YOLOLabels

def ConvertTrajToBoundingBoxes(im,length=128,times=128,treshold=0.5,trackMultiParticle=False):
    debug=False

    # Each label has 5 components - image type,x1,x2,y1,y2
    #Labels are ordered as follows: LabelID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM, where 
    #X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
    #Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
    #WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
    #HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
           
    nump = im.shape[-1]-2
    batchSize = im.shape[0]
    YOLOLabels =np.reshape([None]*batchSize*nump*5,(batchSize,nump,5)) #np.zeros((batchSize,nump,5))#np.reshape([None]*1*2*5,(1,2,5))#
    for j in range(0,batchSize):
        for k in range(0,nump):
            particle_img = im[j,:,:,2+k]
            particleOccurence = np.where(particle_img>treshold)
            if np.sum(particleOccurence) <= 0:
                pass
            else:
                x1,x2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])  
                y1,y2 = np.min(particleOccurence[0]),np.max(particleOccurence[0])  

                YOLOLabels[j,k,:] = 0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)         



                if debug:
                    import matplotlib.patches as pch
                    max_nbr_particles = 5
                    nbr_particles = max_nbr_particles
                    plt.figure()
                    ax = plt.gca()
                    plt.imshow(particle_img,aspect='auto')
                    ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
                    plt.imshow(particle_img,aspect='auto')
                    print(YOLOLabels)
                    plt.colorbar()
                    print(str(x1)+"--"+str(x2)+"--"+str(y1)+"--"+str(y2))


        if trackMultiParticle:
            YOLOLabels = YOLOLabelSingleParticleToMultiple(YOLOLabels[0],overlap_thres=0.6,xdim=length,ydim=times) #Higher threshold means more likely to group nearby particles
            if debug:
                plt.figure()
                ax = plt.gca()
                plt.imshow(im[0,:,:,0],aspect='auto')
                YOLOCoords = ConvertYOLOLabelsToCoord(YOLOLabels,xdim=length,ydim=times)
                classes = ['particle','twoparticles','threeparticles']
                colors = ['white','orange','black']
                for p,x1,y1,x2,y2 in YOLOCoords:
                    p = int(p)
                    ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor=colors[p]))   
                    ax.text(x1,y1,(classes[p]),color = colors[p],fontsize=18)

    return YOLOLabels

nump = lambda: 1+np.random.randint(3)#np.clip(np.random.randint(5),0,3)

# Particle params
Int = lambda : 1e-4*(0.1+8*np.random.rand()) #0.1 - 10e-4
Ds = lambda: 0.10*np.sqrt((0.05 + 1*np.random.rand()))#0.10*(0.05 + 1*np.random.rand())#
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
        x0=-1+2*np.random.rand()
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

    TT = int(times/T_reduction_factor)
    LL = int(length/L_reduction_factor)
    nump = nump() # resolve nump for each batch
    batch = np.zeros((batchsize,TT,LL,nump+2))
    
    for b in range(batchsize):

        image = np.zeros((times,length,nump+2))
        
        # Add noise to image
        noise_image, bg0 = gen_noise(image,dX,dA,noise_lev,biglam,bgnoiseCval,bgnoise,bigx0)
        image = generate_trajectories(noise_image,Int,Ds,st,nump)
        
        # Post process
        image = post_process(image,bg0)
        image = skimage.measure.block_reduce(image,(T_reduction_factor,L_reduction_factor,1),np.mean)
        
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

def generate_exp_noise(im,baseNoise):
    times = im.shape[1]
    length = im.shape[2]
    startIndex = np.random.randint(np.size(baseNoise,0)-times)  
    startIndexLength = 0#np.random.randint(np.size(baseNoise,1)-length)  
    baseNoise = baseNoise[startIndex:startIndex+times,startIndexLength:startIndexLength+length]
    baseNoise = baseNoise*(np.random.rand()+0.1)*2
    im = im + np.expand_dims(baseNoise,(0,-1))
    
    return im

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
    def __init__(self, list_path, img_size=128, augment=False, multiscale=False, normalized_labels=True,totalData=10,unet=None,trackMultiParticle=False):
        self.img_files = list_path

        self.label_files = ""
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.totalData = totalData
        self.unet = unet
        self.trackMultiParticle = trackMultiParticle
        self.imSave = np.ones((1,1,1,1))*np.nan
        self.targetSave = np.ones((1,1,1,1))*np.nan
        self.baseNoise = np.tile(np.load('train_val_noise.npy'),5)

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        batchsize = 1
        rotateData = True
        
        times = self.img_size #normal images are 600 x10000
        length = self.img_size
       
        if  np.isnan(self.imSave).any():       
            #Generate image
            if self.img_size==8192:
                length = 128
                times = 128 
                im = create_batch(batchsize,8192,length*4,nump)
                im = skimage.measure.block_reduce(im,(1,64,1,1),np.mean)   
            else:
                im = create_batch(batchsize,times,length*4,nump)
                
            #Generate YOLO boxes
            treshold = 0.5
            if self.img_size ==8192:
                treshold = 0.05 #Downsampling forces us to alter treshold value
            YOLOLabels = ConvertTrajToMultiBoundingBoxes(im,length=length,times=times,treshold=treshold,trackMultiParticle=self.trackMultiParticle)
            
            #If training with U-net, predict images before rotating and saving (for the sake of consistency - maybe change?)
            try:
                im = self.unet.predict(np.expand_dims(im[...,0],axis=-1))   
            except:
                pass
            #Store flipped images and YOLO boxes
            if rotateData:                
                self.imSave = np.zeros((3,times,length,im.shape[-1]))                
                self.imSave[0,:,:,:] = np.flip(im,axis=(1))[0,:,:,:]
                self.imSave[1,:,:,:] = np.flip(im,axis=(2))[0,:,:,:]
                self.imSave[2,:,:,:] = np.flip(im,axis=(1,2))[0,:,:,:]
                
                self.targetSave = np.zeros((3,np.size(YOLOLabels,0),5))  
                if self.img_size==128 and False:
                    self.targetSave = np.zeros((5,np.size(YOLOLabels,0),5))  
                    
                if not np.isnan(np.array(YOLOLabels,dtype=float)).any():
               
                    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,times,length)
                    temp = length-1-flipLabels[:,4] 
                    flipLabels[:,4] = length-1-flipLabels[:,2] 
                    flipLabels[:,2] = temp
                    
                    self.targetSave[0,:,:] = ConvertCoordToYOLOLabels(flipLabels,times,length)
                    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,times,length)
                    temp = times-1-flipLabels[:,3] 
                    flipLabels[:,3] = times-1-flipLabels[:,1] 
                    flipLabels[:,1] = temp 
                    self.targetSave[1,:,:] = ConvertCoordToYOLOLabels(flipLabels,times,length)
                    
                    flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,times,length)
                    temp = length-1-flipLabels[:,4] 
                    flipLabels[:,4] = length-1-flipLabels[:,2] 
                    flipLabels[:,2] = temp    
                    temp = times-1-flipLabels[:,3] 
                    flipLabels[:,3] = times-1-flipLabels[:,1] 
                    flipLabels[:,1] = temp    
                    self.targetSave[2,:,:] = ConvertCoordToYOLOLabels(flipLabels,times,length)
                                                           
                    if self.img_size == 128 and False: #Try "illegal" rotations as well. Currently unused.
                        self.imSave = np.append(self.imSave,np.rot90(im,axes=(1,2),k=1),0) #Rotate by 90, 270 degrees, maybe  ok for square imgs?
                        self.imSave = np.append(self.imSave,np.rot90(im,axes=(1,2),k=3),0)
                        
                        flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,times,length)
                        
                        temp = np.copy(flipLabels[:,2]) 
                        flipLabels[:,2] = times-1-flipLabels[:,1]
                        flipLabels[:,1] = temp
                        
                        temp = np.copy(flipLabels[:,4]) 
                        flipLabels[:,4] = length-1-flipLabels[:,3]
                        flipLabels[:,3] = temp
                        

                        temp = np.copy(flipLabels[:,2])
                        flipLabels[:,2] = np.minimum(flipLabels[:,4],temp)
                        flipLabels[:,4] = np.maximum(flipLabels[:,4],temp)
                        temp = np.copy(flipLabels[:,1])
                        flipLabels[:,1] = np.minimum(flipLabels[:,3],temp)
                        flipLabels[:,3] = np.maximum(flipLabels[:,3],temp)
                        
                        self.targetSave[3,:,:] = ConvertCoordToYOLOLabels(flipLabels,times,length)
                        
                        flipLabels = ConvertYOLOLabelsToCoord(YOLOLabels,times,length)
                        temp = np.copy(flipLabels[:,2]) 
                        flipLabels[:,2] = flipLabels[:,1]
                        flipLabels[:,1] = times-1-temp
                        
                        temp = np.copy(flipLabels[:,4]) 
                        flipLabels[:,4] = flipLabels[:,3]
                        flipLabels[:,3] = times-1-temp
                        
                        temp = np.copy(flipLabels[:,2])
                        flipLabels[:,2] = np.minimum(flipLabels[:,4],temp)
                        flipLabels[:,4] = np.maximum(flipLabels[:,4],temp)
                        temp = np.copy(flipLabels[:,1])
                        flipLabels[:,1] = np.minimum(flipLabels[:,3],temp)
                        flipLabels[:,3] = np.maximum(flipLabels[:,3],temp)
                        self.targetSave[4,:,:] = ConvertCoordToYOLOLabels(flipLabels,times,length)
                        
                else: 
                    self.targetSave =np.reshape([None]*np.size(self.targetSave,0)*5,(np.size(self.targetSave,0),1,5))
        else:
            #Pick out the next image in saved list
            im = np.expand_dims(self.imSave[0,:,:,:],0)
            self.imSave = self.imSave[1:,:,:,:]
            YOLOLabels = self.targetSave[0,:,:]
            self.targetSave = self.targetSave[1:,:,:]
            if not len(self.imSave):          
                self.imSave = np.ones((1,1,1,1))*np.nan
                self.targetSave = np.ones((1,1,1,1))*np.nan
                               

            
        
        if self.unet==None and False: #If images not square, downsample and pad, currently unused
            im = skimage.measure.block_reduce(im,(1,1,4,1),np.mean)
            im = im[:,:,11:139,:]
            padVal = int((times-128)/2)
            im = np.pad(im,((0,0),(0,0),(padVal,padVal),(0,0)))


        try:
            if rotateData is False:
                v1 = self.unet.predict(np.expand_dims(im[...,0],axis=-1))        
            else: 
                v1 = im#np.expand_dims(im[...,1],axis=-1) 
        except:
            v1 = np.expand_dims(im[...,1],axis=-1)
            
            
        if self.img_files == 'train' and self.img_size==128 and np.random.rand()<0.1 and generateNoise:
            v1 = generate_exp_noise(v1,self.baseNoise[0:12000,:])           
       # elif self.img_files=='val' and self.img_size==128 and np.random.rand()<1:
            #v1 = generate_exp_noise(v1,self.baseNoise[8000:-1,:])
            
        # Extract image as PyTorch tensor
        v1 = np.squeeze(v1,0)
        img = transforms.ToTensor()(v1)
        img = torch.cat([img]*3)
        
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
        YOLOLabels = np.array(YOLOLabels,dtype=float)
        
        if not np.isnan(YOLOLabels).any():
            boxes = torch.from_numpy(YOLOLabels).reshape(-1,5)#torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            if not self.normalized_labels:
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
