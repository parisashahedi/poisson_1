# -*- coding: utf-8 -*-
import torch
import numpy as np

pi=float(np.pi)
torch.set_default_dtype(torch.float32)

class Data_Single(object):
    def __init__(self,r0,L,box,device):
        self.r0=r0
        self.L=torch.tensor(L).to(device)
        self.r1=box
        self.device=device

    def sampleDomain_hole(self,num):
        """
        num: the number of training points
        """
        L=self.L
        X=self.sampleFromDomain(num)      
        y=torch.norm(X-L,dim=1)     
        location=torch.where(y<self.r1)[0]
        X_out=X[location,:]      
        
        #location=torch.where(y<self.r0)[0]
        #X_in=X[location,:]      

        return X_out


    def sampleFromDomain(self,num):
        xmin,xmax,ymin,ymax=-self.r1,self.r1,-self.r1,self.r1
        x = torch.rand(10*num,device=self.device).view(-1,1) * (xmax - xmin) + xmin
        y = torch.rand(10*num,device=self.device).view(-1,1) * (ymax - ymin) + ymin
        X = torch.cat((x,y),dim=1)
        y=torch.norm(X-self.L,dim=1)     
        location=torch.where(y<self.r1)[0]
        X=X[location,:] 
        X=X[:num]

        return X


    def sampleFromBoundary(self,num):

        L=self.L
        psi=2*pi*torch.rand(num,device=self.device).view(-1,1)   #[0,2pi]
        return psi.to(self.device)

