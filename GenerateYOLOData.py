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
    v1 = np.zeros((batchsize,nump,times,length))#,dtype="float32")

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

def ClusterSeparateParticles(traj):
    traj = v1.T

    plt.imshow(np.sum(traj[...,0],-1),aspect='auto')
    plt.show()
    import skimage.morphology
    from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
    
    X,Y=np.meshgrid(np.linspace(-1,1,traj.shape[1]),np.linspace(-1,1,traj.shape[0]))
    ints=[]
    diffs=[]
    traj = np.sum(traj,-2)
    for ii in range(traj.shape[-1]):
        binary=np.zeros(traj[:,:,ii].shape)
    
        binary[traj[:,:,ii]>0.8]=1
        #plt.imshow(binary,aspect='auto')
        #binary2=skimage.morphology.dilation(binary,np.ones((20,15)))
        #plt.imshow(binary2,aspect='auto')
        from skimage import measure
        blobs_labels = measure.label(binary, background=0)
        #plt.imshow(blobs_labels,aspect='auto')
        #plt.show()
        binary=np.reshape(binary,(traj.shape[0]*traj.shape[1],))
        blobs_labels=np.reshape(blobs_labels,(traj.shape[0]*traj.shape[1],))
        data=np.reshape(traj[ii,:,:,2],(traj.shape[0]*traj.shape[1],))
        #datac=np.reshape(traj[ii,:,:,-1],(traj.shape[0]*traj.shape[1],))
        #dataI=np.reshape(traj[ii,:,:,0],(traj.shape[0]*traj.shape[1],))
        dataD=np.reshape(traj[ii,:,:,1],(traj.shape[0]*traj.shape[1],))
    
        datareshaped=np.reshape(p3[ii],(p3.shape[1]*p3.shape[2],7))
        for i in range(np.amax(blobs_labels)):
           
            datainds=np.where(blobs_labels==i+1)[0]
    
            ns=np.sum(data[datainds]*datac[datainds])
            #print(ns)
            if ns>.5 and ns<1.5:
                ints=np.append(ints,np.mean(dataI[datainds]))
                diffs=np.append(diffs,np.mean(dataD[datainds]))
            elif ns>1.5:
                ns=np.round(ns)
    
                reldat=np.zeros((len(datainds),4))
                reldat[:,0]=datareshaped[datainds,0]
                reldat[:,1]=datareshaped[datainds,1]
                reldat[:,3]=datareshaped[datainds,5]
                vs=reldat[:,3]
    
                Ts=np.reshape(X,(p3.shape[1]*p3.shape[2],))
                Xs=np.reshape(Y,(p3.shape[1]*p3.shape[2],))
                Ts=Ts[datainds]
                Xs=Xs[datainds]
                x0s=Xs-vs*Ts
                reldat[:,2]=x0s
                centroid, label = kmeans2(reldat, int(ns), minit='++')
                for ij in range(int(ns)):
                    ints=np.append(ints,centroid[ij,0])
                    diffs=np.append(diffs,centroid[ij,1])
    
    plt.scatter(diffs,ints)
#%% Generate validation data batches
if __name__ == "__main__":
    print_labels = False
    dx_real = 0.03
    dt_real = 0.005
    
    val_bsize = 3
    Int = lambda: 1.05e-3*(0.08+0.8*np.random.rand())
    Ds = lambda: 0.10*np.sqrt((0.05 + 1.0*np.random.rand()))
    st = lambda: 0.04 + 0.01*np.random.rand()
    
    I_max = 1.05e-3*0.88
    D_max = 0.10*np.sqrt(1.05)
    # print("I_max = {}".format(I_max))
    # print("D_max = {}".format(D_max))
    # print()
    
    length = 512
    times = 128
    
    dA = lambda: 0.00006 * (0.7 + np.random.rand())
    dX = lambda: 0.00006* (0.7 + np.random.rand())
    bgnoiselev = lambda: 0.0006* (0.7 + np.random.rand())
    
    #dA = 0
    #dX = 0
    #bgnoiseelev = lambda: 0
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
                                                     batchsize=val_bsize,
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
    #preds = m.predict(np.expand_dims(val[...,0],axis=-1))
    
    print('Validation data generated.')

#%% Plot a particle and its mask/true trajectory
if __name__ == "__main__":
    plt.close('all')
    valL[1].shape
    j = 0
    mask = valL[1][j,...,2]
    plt.imshow(val[j,...,0].T,aspect='auto')
    #plt.colorbar()
    
    plt.figure()
    plt.imshow(mask.T,aspect='auto')
    
    plt.figure()
    p1 = v1[j,0,...]
    p2 = v1[j,1,...]
    particles = p1+p2
    plt.imshow(particles.T,aspect='auto')
    #plt.imshow(p2.T,aspect='auto')

#%% 
if __name__ == "__main__":
    plt.close('all')
    import matplotlib.patches as pch
    treshold = 0.5 #A particle exists when the probability of its existence is this treshold
    for j in range(0,val_bsize):
        particles=np.zeros((times,length))
        plt.figure(j)
        for k in range(0,nump):
            p = v1[j,k,...]
            particleOccurence = np.where(p>treshold)
            x1,x2 = particleOccurence[0][0],particleOccurence[0][-1]
            y1,y2 = np.min(particleOccurence[1]),np.max(particleOccurence[1])
            particles +=p
            ax = plt.gca()
            ax.add_patch(pch.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,zorder=2,edgecolor='white'))
    
        plt.imshow(particles.T,aspect='auto')
    
            