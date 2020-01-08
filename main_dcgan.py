# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:33:18 2019

@author: eric
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import multiprocessing
import model_dcgan
import copy
import matplotlib.pyplot as plt


def imbalanced_data(dataset,data_loader, cls_num,ratio,n_cpu,batch_size):

    cls_num=cls_num.split('_')
    idx_candi=np.array([]).astype(int)

    for num_candi in cls_num:
        if opt.dataset == 'MNIST':
            candi=np.where(data_loader.dataset.targets.numpy() == int(num_candi))[0]

        elif opt.dataset == 'cifar10':
            candi=np.where(torch.as_tensor(data_loader.dataset.targets).numpy() == int(num_candi))[0]
            
        idx_candi=np.append(idx_candi,candi)

    print(idx_candi)

    num_of_choice=int(len(idx_candi)*ratio)
    print('num_of_choice :', num_of_choice)
    
    idx_=np.random.choice(idx_candi,num_of_choice,replace=False)
    new_dataset=copy.deepcopy(dataset)
    
    if opt.dataset == 'MNIST':
        new_dataset.targets = torch.from_numpy(data_loader.dataset.targets.numpy()[idx_])
        new_dataset.data = torch.from_numpy(data_loader.dataset.data.numpy()[idx_])

    elif opt.dataset == 'cifar10':
        new_dataset.targets = torch.as_tensor(data_loader.dataset.targets).numpy()[idx_]
        new_dataset.data = torch.as_tensor(data_loader.dataset.data).numpy()[idx_]
            
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu,drop_last=True)
    
    if opt.dataset == 'MNIST':
        num_list=[len(np.where(new_data_loader.dataset.targets.numpy()==x)[0]) for x in range(10) ]
    
    elif opt.dataset =='cifar10':
        num_list=[len(np.where(torch.as_tensor(new_data_loader.dataset.targets).numpy()==x)[0]) for x in range(10) ]
    
    print('num_of_sample_class :', num_list)
    print('ratio_of_sample_class :', np.around(np.array(num_list)/len(dataset),2))
    return new_dataset,new_data_loader

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='cifar10 | mnist', default='cifar10')
    parser.add_argument('--dataroot', help='path to dataset', default='./im_data')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    #parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cls_num', default='3_4',help='list_of_numbers')
    parser.add_argument('--imb_ratio', type=float,default=0.3,help='imb_ratio')
    
    opt = parser.parse_args()
    print(opt)
    
    n_cpu= multiprocessing.cpu_count()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opt.cls_num)
    cudnn.benchmark = True
    
    if opt.dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    
    
    elif opt.dataset == 'MNIST':
        train_dataset = dset.MNIST(root=opt.dataroot, download=True, train=True, transform=transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])]))
    
    
    assert train_dataset
    
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=n_cpu,drop_last=True)
    
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    ngpu=1
    
    if opt.dataset == 'MNIST':
        nc = 1
        nb_label = 10
    else:
        nc = 3
        nb_label = 10
    
    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    Generator=model_dcgan.Generator
    Discriminator=model_dcgan.Discriminator
        
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.apply(weights_init)
   
    netD = Discriminator(ngpu, nc,ndf).to(device)
    netD.apply(weights_init)

    
    
    #netG = torch.nn.DataParallel(model_dcgan.Generator(nz, ngf, nc), device_ids=[0,1])
    #netD = torch.nn.DataParallel(model_dcgan.Discriminator(ndf, nc, nb_label), device_ids=[0,1])

    s_criterion = nn.BCELoss()
    c_criterion = nn.NLLLoss()
    
    if opt.dataset =='MNIST':
        input = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
    else:
        input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        
        
    real_label = 1
    fake_label = 0
    
    netD.to(device)
    netG.to(device)
    s_criterion.to(device)
    c_criterion.to(device)
    
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def test(predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)
    
    
    time_start=0

    imbalanced_dataset,imbalanced_data_loader=imbalanced_data(train_dataset, dataloader, opt.cls_num, opt.imb_ratio, n_cpu, opt.batchSize)
    
    c_time=0
    
    fixed_noise = torch.randn(128, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    
    niter = 100
    g_loss = []
    d_loss = []
    
    
    for epoch in range(niter):
        for i, data in enumerate(imbalanced_data_loader, 0):
           
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)
    
            output = netD(real_cpu)
            errD_real = s_criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
    
            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = s_criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = s_criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
    
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            
            Loss_D=[]
            Loss_G=[]
            #plt.figure()
            #plt.plot(range(10),Loss_D,color='blue')
            #plt.plot(range(10),Loss_G,color='red')
            #plt.legend(['Discriminator Loss','Genearator Loss'],loc='upper right')
            #save the output
            if i % 100 == 0:
                print('saving the output')
                vutils.save_image(real_cpu,r'./output_im/real_samples.png',normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),r'./output_im/fake_samples_epoch_%03d.png' % (epoch),normalize=True)
        
        # Check pointing for every epoch
        torch.save(netG.state_dict(), r'./weights_im/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), r'./weights_im/netD_epoch_%d.pth' % (epoch))
        plt.plot(range(100),trn_lstm_loss,color='green')
        plt.plot(range(100),val_lstm_loss,color='red')
        plt.legend(['Discriminator Loss','Generator Loss'],loc='upper right')
        
