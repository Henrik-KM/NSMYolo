import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
def gen_noise(bgnoiselev,dA,vibamp,f2,x,times,length): #Generating simulated noise
    bgnoiseCval=.03+.02*np.random.randn()
    bgnoise=(.003+.005*np.random.rand())*np.random.randn(length)
    #vibamp=dX#np.random.rand()*dX
    biglam=np.random.rand()*.2+.6
    bigx0=np.random.randn()*.2
    tempcorr=3*np.random.rand()
    dAmp=dA()#*np.random.rand()
    shiftval=vibamp*np.random.randn()
    dx=0
    dx2=0
    dAmp0=0
    bg0=f2(1,bigx0,biglam,0,x)
    noise=np.zeros((times,length))
    for j in range(times):
        dx=np.sqrt(1-np.exp(-2/tempcorr))*vibamp*np.random.randn()+np.exp(-1/tempcorr)*dx
        dx2=np.sqrt(1-np.exp(-2/tempcorr))*vibamp*np.random.randn()+np.exp(-1/tempcorr)*dx2

        bgnoiseC=f2(1,0,bgnoiseCval,dx2,x)
        #bgnoiseC2=f2(1,0,bgnoiseCval,dx2+shiftval,x)
        bgnoiseC/=np.sum(bgnoiseC)
        bg=f2(1,bigx0+dx,biglam,0,x)+convolve(bgnoise,bgnoiseC,mode="same")
        #bg2=f2(1,bigx0+dx2+shiftval,biglam,0,x)+convolve(bgnoise,bgnoiseC2,mode="same")
        dAmp0=np.sqrt(1-np.exp(-2/tempcorr))*dAmp*np.random.randn()+np.exp(-1/tempcorr)*dAmp0
        bg*=(1+dAmp0)
        noise[j,:]=bg
    return noise,bg0

def gen_noise_experimental(bgin,dA,vibamp,times): #Generating noise from experimental data

    bginflipped=np.flip(bgin)
    bgin=np.append(2*bgin[0]-bginflipped,bgin)
    bgin=np.append(bgin,2*bgin[-1]-bginflipped)
    x=np.linspace(-3,3,512*3)
    kx=2*np.pi/(3*(x[1]-x[0]))*x
    fftv0=np.fft.fftshift(np.fft.fft(bgin))
    dx=0
    noise=np.zeros((times,3*512))+0j
    shiftval=vibamp*np.random.randn()
    dAmp=dA
    dAmp0=0
    tempcorr=3*np.random.rand()

    for i in range(times):
        dx=np.sqrt(1-np.exp(-2/tempcorr))*vibamp*np.random.randn()+np.exp(-1/tempcorr)*dx
        fftv=fftv0*np.exp(1j*kx*dx)
        dAmp0=np.sqrt(1-np.exp(-2/tempcorr))*dAmp*np.random.randn()+np.exp(-1/tempcorr)*dAmp0
        fftv*=(1+dAmp0)
        noise[i,:]=np.fft.fftshift(fftv)
    noise=np.real(np.fft.ifft(noise,axis=-1))
    noise=noise[:,512:1024]
    return noise,np.mean(noise,axis=0)


#Generate batches for training a network
import numpy as np

from scipy.special import j1
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as imrotate
from timeit import default_timer as timer
from scipy.interpolate import RectBivariateSpline
from scipy.signal import convolve, convolve2d
import scipy.io as IO
import skimage.measure as measure
from scipy import stats
import skimage

def Generate8b8batchboxesv2Generator(bgnoiselev,Int=lambda: 1e-4,st=lambda:0.05,Ds=lambda: 0.05,
                                     dA=0,dX=0,batchsize=8,length=512,times=64,bgexp=None,
                                     print_labels=False, time_reduction=1, length_reduction=1,
                                     downsampling_factor_for_length=1,nump=2):
    x=np.linspace(-1,1,length)
    t=np.linspace(-1,1,times)
    t_red = time_reduction # downsampling in time direction
    l_red = length_reduction # downsampling in length direction
    t2=np.linspace(-1,1,times//t_red)
    x2=np.linspace(-1,1,length//l_red)
    G=lambda a,b,x0,s,x: a*np.exp(-(x-x0)**2/s**2)+b
    f2=lambda a,x0,s,b,x: a*np.exp(-(x-x0)**2/s**2)+b
    dx_real = 0.03
    dt_real = 0.005
    #Define number of particles in the simulation
    v1 = np.zeros((batchsize,nump,times,length))

    while True:
        batch=np.zeros((batchsize,times,length,2))
        label=np.zeros((batchsize,times//t_red,length//l_red,7))


        ld=np.zeros((batchsize,2))
        X, Y=np.meshgrid(np.linspace(-1,1,times),x)
        X2, Y2=np.meshgrid(np.linspace(-1,1,times//t_red),x2)
        Y2s=np.repeat(np.expand_dims(np.transpose(Y2),axis=-1),times,axis=-1)
        X2s=np.repeat(np.expand_dims(np.transpose(X2),axis=-1),times,axis=-1)

        counts=np.zeros((batchsize,))
        no=np.zeros((batchsize,))
        vibamp=dX()
        DList=np.zeros((batchsize,))
        i=-1
        while i< batchsize-1:
            i+=1

            #print(i)
            counts=np.zeros((nump,))

            x0=np.zeros((times,nump))
            I=np.zeros((nump,))
            D=np.zeros((nump,))
            s0=np.zeros((nump,))
            for ij in range(nump): #Give properties to each particle
                I[ij]=Int()
                D[ij]=Ds()
                s0[ij]=st()
                D_factor = 256**2/100*dx_real**2/dt_real/2/57
                I_factor = s0[ij]*np.sqrt(np.pi)*256*dx_real                
                I_max = 1.05e-3*0.88
                D_max = 0.10*np.sqrt(1.05)
                if print_labels:
                    print("Input I (%): {}".format(I[ij]/I_max))
                    print("Input D (%): {}".format(D[ij]/D_max))
                #print("Input I: {}".format(I[ij]*I_factor*2000))
                #print("Input D: {}".format((D[ij]*10)**2*D_factor))
            vel=10*(np.random.rand(nump)-.5)/length #Average velocities of the particles

            t0=(1.1*np.random.rand(nump)-.1)*times #Time point at which particle reaches x=-1
            x0[0,:]=-vel*t0-1 #Starting point of particle
            x0[0,:]=-1+2*np.random.rand(nump)
            #for ij in range(nump):
            # ld[i,0]=10*D[ij]
            # ld[i,1]=.2*I[ij]*s0[ij]**2/.05**2 #Make a list of "ground truth" values
            #Initialize stuff...
            dists=np.zeros((*label.shape[1:-1],nump))
            vels=np.zeros((*label.shape[1:-1],nump))
            DList[i]=10*D[0]
            diffs=np.zeros((*label.shape[1:-1],nump))
            ints=np.zeros((*label.shape[1:-1],nump))
            v2=np.zeros((*label.shape[1:-1],nump))
            xline=np.zeros(x0.shape)
            slope=np.zeros((nump,))
            inter=np.zeros((nump,))
            maxdist=np.zeros((nump,))
            mx=np.zeros((*label.shape[1:-1],nump))
            cs=np.zeros((*label.shape[1:-1],nump))
            dx=0
            dx2=0


            if i%16==0:
                if bgexp is not None:
                    bg,bg0=gen_noise_experimental(bgexp,dA,vibamp,times)#Generate experimental noise, based on input noise profile bgexp
                else:
                    bg,bg0=gen_noise(bgnoiselev, dA,vibamp,f2,x,times,length) #Generate simulated noise
                batch[i,:,:,1]=bg

            else:
                bg=batch[i-1,:,:,1]
                batch[i,:,:,1]=bg
            bg=batch[i,:,:,1]

            value=bg
            tmp=np.zeros(label.shape[1:-1])
            for k in range(nump):

                x0[:,k]=x0[0,k]+np.cumsum(vel[k]+D[k]*np.random.randn(times)) #Simulate a Brownian particle
                inds=np.where(np.abs(x0[:,k])<1)[0] #Find where the particle is inside channel
                print(inds.shape)


                if len(inds)>2:
                    sl, intercept, r_value, p_value, std_err = stats.linregress(t[inds],x0[inds,k])#Line regress over particle positions inside channel
                    counts[k]=len(inds)
                else:
                    sl, intercept, r_value, p_value, std_err = stats.linregress(t,x0[:,k])#If inside only a single or two pixels, make a line regress over entire trajectory (will not matter anyway)
                    counts[k]=1
                    
                # Store slope, intercept and the fitted line for each particle
                slope[k]=sl#
                inter[k]=intercept
                xline[:,k]=slope[k]*t+intercept

                if len(inds)>2:
                    maxdist[k]=np.amax(np.abs(x0[inds,k]-xline[inds,k]))#Find max excursion from line

                value=value*np.transpose(f2(-I[k],x0[:,k],s0[k],1,Y)) #Add particle to noise image

                # Make profile for labels
                v1[i,k,:,:] = np.transpose(f2(1,x0[:,k],.1,0,Y))
                v2[:,:,k]=skimage.measure.block_reduce(v1[i,k,:,:], (t_red,l_red), np.max)
                blobs_labels = measure.label((v2[:,:,k]>.3).astype('float32'), background=0) #Segment the segmented image into separate connected components (If the same particle moves in and out of the channel multiple times, this should be split into several particles)

                for kj in range(1,1+np.amax(blobs_labels)):
                    inds=np.where(blobs_labels==kj)
                    bg=np.zeros(blobs_labels.shape)
                    bg[inds[0],inds[1]]=1
                    tmp[inds[0],inds[1]]=1
                    cs[:,:,k]+=bg/np.sum(bg+1e-7) #Particle count channel

                dis=np.sqrt((Y2s-xline[:,k])**2+(X2s-t)**2)

                dists[:,:,k]=np.amin(dis,axis=-1) #Find distance to line from each pixel
                mx[:,:,k]=np.argmin(dis,axis=-1)
                vels[:,:,k]=3*vel[k]
                diffs[:,:,k]=(10*D[k])**2*256**2/100*dx_real**2/dt_real/2
                ints[:,:,k]=I[k]*s0[k]*np.sqrt(np.pi)*256*dx_real # *dx
#                 if print_labels:
#                     print("I(input): {}".format(diffs[0,0,0]))
#                     print("D(input): {}".format(ints[0,0,0]))

            label[i,:,:,2]=tmp#Assign segmentation channel
            dist2=np.argmin(np.abs(dists),axis=-1) #Find which particle is closest to each pixel
            for kl in range(label.shape[1]):
                for mn in range(label.shape[2]):
                    #Assign properties of correct particle to each pixel
                    label[i,kl,mn,0]=diffs[kl,mn,dist2[kl,mn]]*label[i,kl,mn,2]
                    label[i,kl,mn,1]=ints[kl,mn,dist2[kl,mn]]*label[i,kl,mn,2]
                    ti=t2[kl]
                    xi=slope[dist2[kl,mn]]*ti+inter[dist2[kl,mn]]
                    dx=x2[mn]-xi
                    label[i,kl,mn,3]=dx*label[i,kl,mn,2]
                    label[i,kl,mn,4]=slope[dist2[kl,mn]]*label[i,kl,mn,2]/10
                    label[i,kl,mn,5]=maxdist[dist2[kl,mn]]*label[i,kl,mn,2]

            value*=(1+bgnoiselev()*np.random.randn(times,length))# Add white noise to image
            label[i,:,:,6]=np.sum(cs,axis=-1) # Assign particle count channel
            batch[i,:,:,0]=value


            batch[i,:,:,0]/=bg0#Normalize image by the bare signal
            batch[i,:,:,0]-=np.expand_dims(np.mean(batch[i,:,:,0],axis=0),axis=0)#Subtract mean over image

            # Perform same preprocessing as done on experimental images
            ono=np.ones((200,1))
            ono[0:80]=1
            ono[120:]=1
            ono=ono/np.sum(ono)
            ono2=np.ones((1,50))
            ono2/=np.sum(ono2)
            batch[i,:,:,0]-=convolve2d(batch[i,:,:,0],ono,mode="same")
            batch[i,:,:,0]-=convolve2d(batch[i,:,:,0],np.transpose(ono),mode="same")

            # Normalize by standard deviation of noise
            a=np.expand_dims(np.std(batch[i,:,:,0],axis=1),axis=1)
            no[i]=np.mean(a)
            
            batch[i,:,:,0]-=np.expand_dims(np.mean(batch[i,:,:,0],axis=0),axis=0)
            batch[i,:,:,0]*=1000 # *1000 sets input in ca (0,1)-range

            label[i,:,:,0]*=1/57 # set D in (0,1)-range         
            label[i,:,:,1]*=2000 # set I in (0,1)-range    
            
            ld[i,1]=np.sum(label[i,:,:,1]*label[i,:,:,2])/(1e-3+np.sum(label[i,:,:,2]))
            ld[i,0]=np.sum(label[i,:,:,0]*label[i,:,:,2])/(1e-3+np.sum(label[i,:,:,2]))
            
            if print_labels:
                print('I-label: (true_value){}'.format(ld[i,1]/2000))
                print('D-label (true_value): {}'.format(ld[i,0]*57))
                print()

        # Downsample simulated image
        DFL = downsampling_factor_for_length
        batch=skimage.measure.block_reduce(batch, (1,1,DFL,1), np.mean)


        yield batch[...,:1],[ld[:,:2],label[...,:3]],v1#, DList, counts/(label.shape[1]*label.shape[2])*40, no
        
def ConvertTrajToBoundingBoxes(v1,batchSize,nump,length=512,times=128,treshold=0.5):
    YOLOLabels = np.zeros((batchSize,nump,5)) # Each label has 5 components - image type,x1,x2,y1,y2
    #Labels are ordered as follows: LabelID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM, where 
    #X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
    #Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
    #WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
    #HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
    try:
        for j in range(0,batchSize):
            for k in range(0,nump):
                p = v1[j,k,...]
                particleOccurence = np.where(p>treshold)
                if not np.size(particleOccurence)  == 0:
                    x1,x2 = particleOccurence[0][0],particleOccurence[0][-1]
                    y1,y2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])
                    
                    YOLOLabels[j,k,:] = 0, np.abs(x2+x1)/2/(times-1), (y2+y1)/2/(length-1),(x2-x1)/(times-1),(y2-y1)/(length-1)
                else:
                    YOLOLabels = np.delete(YOLOLabels,[j,k],1)
    except:
        print("Label generation failed. Continuing..")

    return YOLOLabels


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
    def __init__(self, list_path, img_size=416, augment=False, multiscale=False, normalized_labels=True,totalData=1000):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.totalData = totalData

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        print_labels = False
        
        batchSize = 1
        Int = lambda: 1.05e-3*(0.08+0.8*np.random.rand())
        Ds = lambda: 0.10*np.sqrt((0.05 + 1.0*np.random.rand()))
        st = lambda: 0.04 + 0.01*np.random.rand()
        
    
        length = 512
        times = 128
        
        dA = lambda: 0.00006 * (0.7 + np.random.rand())
        dX = lambda: 0.00006* (0.7 + np.random.rand())
        bgnoiselev = lambda: 0.0006* (0.7 + np.random.rand())
    
        time_reduction = 16
        length_reduction = 64
        downsampling_factor_for_length = 1
        nump=2 #number of particles
        data_generator = Generate8b8batchboxesv2Generator(bgnoiselev=bgnoiselev,
                                                         Int=Int,
                                                         st=st,
                                                         Ds=Ds, 
                                                         dA=dA,
                                                         dX=dX,
                                                         batchsize=batchSize,
                                                         length=length,
                                                         times=times,
                                                         bgexp=None,
                                                         print_labels=print_labels,
                                                         time_reduction=time_reduction,
                                                         length_reduction=length_reduction,
                                                         downsampling_factor_for_length=downsampling_factor_for_length,
                                                         nump=nump)
        val,valL,v1 = next(data_generator)
        
        # For training on iOC = 5e-4, D = [10,20,50] mu m^2/s
        # Range on Ds: 0.03 -> 0.08
        # Range on Is: 5e-3 = good contrast
        
        # Plot predictions of validation samples
        val=val[...,:1]
        valld = valL[0]
        
        YOLOLabels=ConvertTrajToBoundingBoxes(v1,batchSize,nump,length=512,times=128,treshold=0.5)
       # v1 = np.sum(v1[0,...],0).T #Place all particles in the same image
        v1 = np.sum(v1,1).T
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(v1)
        img = torch.cat(3*[img]) # Convert to 3-channel image to simulate RGB information

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

        return "", img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return self.totalData
