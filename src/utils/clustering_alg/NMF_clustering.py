#/--coding:utf-8/
#/author:Ethan Wang/

import numpy as np
import sys
import scipy.io
import copy
import math
import os

def matrix_mean(V, segment_width, f_dim):

    for run in range(0, abs(segment_width)):
        temp[:,:,run]=V[(run)*f_dim:(run+1)*f_dim, run:end-segment_width+run]
    V=np.mean(temp, axis=2)

    return V

def matrix_standardization(data):

    baseline = np.min(data)
    range_ = np.max(data)-baseline
    aa=np.full((data.shape[0],data.shape[1]),baseline)
    data = np.subtract(data,aa)/range_

    return data,aa,range_

def basis_exchange(W, H, k_range, segment_width):

    period_feature=np.array([[]])
    for n in range(0, W.shape[1]):
        aa = np.fft.fft(H[n,:] - np.mean(H[n,:]))#/H.shape[0]
        aa = np.abs(aa[1:np.int(np.floor(aa.shape[0]/2+1))])
        if(n==0):
            period_feature = np.zeros((aa.shape[0], W.shape[1]))
            diff_period = np.zeros((W.shape[1],1))
        period_feature[:,n] = aa
        diff_period[n,0] = np.max(aa[1:]) - np.median(aa)

    period_feature = period_feature[1:,:] 
    class_index = nmfsc_clustering(period_feature, k_range)
    class_index = np.reshape(class_index,(-1, ))

    return W, H, class_index, period_feature

def sequential_matrix( data, segment_width):

    y = data.shape[0]
    x = data.shape[1]
    if(segment_width>=1):
        output = np.zeros((y*segment_width, x+segment_width-1))
        for n in range(0,segment_width):
            output[n*y:(n+1)*y  ,n:output.shape[1]+1-segment_width+n]=data;

    return output

def projfunc(s, k1, k2, nn):
    N = len(s)
    if(not(nn)):
        isneg=np.all(s<0) 
        s = abs(s)
    v = s+(k1-sum(s))/N
    zerocoeff = np.array([])
    j=0
    while(1):
        midpoint = np.dot(np.ones((N,1)),k1)/(N-len(zerocoeff))
        midpoint[zerocoeff] = 0
        w = v-midpoint
        a = np.sum(w**2)
        b = np.dot(2*np.transpose(w),v)
        c = np.sum(v**2)-k2
        alphap = (-b+math.real(math.sqrt(b**2-4*a*c)))/(2*a)
        if(np.all(v>=0)):
            usediters = j+1
            break
        j=j+1
        zerocoeff = np.where(v<=0)
        for i in range(0,len(zerocoeff)):
            v[zerocoeff[i]]=0
        tempsum = np.sum(v)
        v = v+(k1-tempsum)/(N-len(zerocoeff))
        for i in range(0,len(zerocoeff)):
            v[zerocoeff[i]]=0
    if(not(nn)):
        v = ((-2)*isneg+1)*v

    return v, usediters
 
def nmfsc_clustering(data, k_range, sH=0.3, iter_num = 500, replica = 10):

    if(len(k_range)==1):
        replica = 1
        consensus_matrix = np.array([])
        dispersion = np.array([])
    data_backup = data
    #initialize the clustering
    for n in range(0, len(k_range)):
        k = k_range[n]
        for m in range(0, replica):
            W, H = nmfsc(data, k, np.array([]), sH, iter_num, 0, [], [])
            c = np.argmax(H, axis=0)
    class_index = c

    return class_index
    
def nmfsc( V, rdim, sW, sH, iter_num, showflag, W0, H0):
    
    V = V/np.max(V)
    vdim = V.shape[0]
    samples = V.shape[1]
    
    W = np.absolute(np.random.randn(vdim, rdim))
    H = np.absolute(np.random.randn(rdim, samples))
    H = H/np.dot(np.reshape(np.sqrt(np.sum(H**2, 1)),(H.shape[0], 1)),np.ones((1,samples)))
    if(not(np.all(sW))):
        L1a = math.sqrt(vdim)-(math.sqrt(vdim)-1)*sW
        for i in range(0, rdim):
            W[:,i] = projfunc(W[:,i],L1a,1,1)
    if(not(np.all(sH))):
        L1s = math.sqrt(samples)-(math.sqrt(samples)-1)*sH
        for i in range(0, rdim):
            H[i,:] = projfunc(H[i,:],L1s,1,1)
    objhistory = np.array([0.5*np.sum(np.sum(np.subtract(V, np.dot(W,H))**2))])
    #initial step
    stepsizeW = 1
    stepsizeH = 1
    

    for iteration in range(0, iter_num):
        Wold = np.array(W)
        Hold = np.array(W)
        ##update
        if(not(np.all(sH))):
            dH = np.dot(np.transpose(W), np.subtract(np.dot(W,H),V))
            begobj = objhistory[-1];
            count = 1
            while(1):
                Hnew = H - np.dot(stepsizeH, dH)
                for i in range(0,rdim):
                    Hnew[i,:] = np.transpose(projfunc(np.transpose(Hnew[i,:]), L1s, 1, 1))
                    
                newobj = 0.5*np.sum(np.sum(np.substract(V, np.dot(W,Hnew))**2))
                if(newobj<=begobj):
                    break
                else:
                    count= count+1
                    if(count>=10):
                        break
                stepsizeH = stepsizeH/2
            stepsizeH = stepsizeH*1.2
            H = Hnew
        else:
            H = H*(np.dot(np.transpose(W),V))/(np.dot(np.dot(np.transpose(W),W),H)+1e-9)
            norms = np.sqrt(np.sum(np.transpose(H)**2))
            #print(norms.shape)
            H = H/(np.dot(np.transpose(norms),np.ones((1,samples))))
            W = W*(np.dot(np.ones((vdim,1)),norms))
        #update W
        if(not(np.all(sW))):
            dW = np.dot(np.subtract(np.dot(W,H), V), np.transpose(H))
            begobj = 0.5*sum(sum(np.dot(np.subtract(V, W), H)**2))

            count = 1
            while(1):
                Wnew = np.subtract(W, np.dot(stepsizeW, dW))
                norms = math.sqrt(np.sum(Wnew**2))
                for i in range(0, rdim):
                    Wnew[:,i] = projfunc(Wnew[:,i], np.dot(L1a, norms[i]), np.pow(norms[i], 2),1)
                newobj = 0.5*sum(sum(np.subtract(V, np.dot(Wnew, H)**2)))
                if(newobj<=begobj):
                    break
                else:
                    count= count+1
                    if(count>=10):
                        break
                    stepsizeW = stepsizeW/2
                stepsizeW = stepsizeW*1.2
                W = Wnew
        else:
            W = W*(np.dot(V, np.transpose(H)))/((np.dot(np.dot(W,H), np.transpose(H)))+1e-9)
        newobj = 0.5*np.sum(np.subtract(V, np.dot(W,H))**2)
        newo = np.array([newobj])
        objhistory = np.concatenate((objhistory, newo))

    return W, H

