# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:16:41 2019

@author: User
"""

import torch
from model_dcgan import Generator, Discriminator
from matplotlib.pyplot import imshow, imsave
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import argparse

def get_sample_single_image(G, counts ,n_noise,DEVICE,file_path,save):
    #torch.radn(a,b) => b 크기의 데이터 a개를 만들어라
    z = torch.randn(counts, n_noise, 1, 1).to(DEVICE)
    y_hat = torch.squeeze(model(z), 1) # (100, 28, 28)
#    print(y_hat)
#    result = (y_hat.cpu().data.numpy()*255).astype(int)
#    print(result.shape)
#    print(type(result))
#    print(result)

            
            
    if save == True:
        for index in range(y_hat.shape[0]):
            new_image=y_hat.data[index]
            save_image(new_image,file_path+'/%s_%s.png'%(index,1))
            print(index)
            #print(new_image)
        
    return ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='MNIST')
    parser.add_argument('--nc', type=int, help='MNIST 1, cifar10 3')
    parser.add_argument('--ngf', type=int,default=32, help="path to netG (to continue training)")
    parser.add_argument('--counts', type=int)
    parser.add_argument('--model_path', help='path_of_model')
    parser.add_argument('--save', type=bool)
    opt = parser.parse_args()
    
    
    path = r'C:\Users\eric\Desktop\gan_code\weights_im\netG_epoch_399.pth'
#    path=r'.\models\G.pkl'

    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_noise=100 
    
    model=Generator(ngpu=1, nz=100, ngf=64, nc=3)
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    file_path=r'./oversample'
    if not os.path.exists(file_path):   
        os.makedirs(file_path)
    
    counts=100
    get_sample_single_image(model, counts, n_noise,DEVICE,file_path,True)


