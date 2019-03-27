
from sklearn.metrics import mean_squared_error
from skimage import measure

import numpy as np
import scipy.io as sio
import math

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.autograd import Variable

from .utils import TrdataLoader
from .utils import TedataLoader

from .dncnn import DnCNN

class Noise2Noise(object):
    def __init__(self,_date, _tr_data_dir=None, _te_data_dir=None, _g_weight_dir=None, _training_type='clean', _noise_dist=25, _epochs=50, _mini_batch_size=64, _learning_rate=0.001, _model_name='DnCNN', _case = None, _generator = None, _avg_size = 4, _crop_size = 100):
        self.tr_data_dir = _tr_data_dir
        self.te_data_dir = _te_data_dir
        self.g_weight_dir = _g_weight_dir
        self.training_type = _training_type
        self.noise_dist = _noise_dist
        self.mini_batch_size = _mini_batch_size
        self.learning_rate = _learning_rate
        self.model_name = _model_name
        self.epochs = _epochs
        self.avg_size = _avg_size
        
        self.img_size = _crop_size
        
        _transforms_tr = [ transforms.RandomCrop(self.img_size), 
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        ]
        
        _transforms_te = [ transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,)) ]
        
        self.transform_tr = transforms.Compose(_transforms_tr)
        self.transform_te = transforms.Compose(_transforms_te)
        
        self.G = _generator
        self.tr_data_loader = TrdataLoader(self.tr_data_dir, self.training_type, self.g_weight_dir, self.noise_dist, self.transform_tr, self.G, self.avg_size)
        self.te_data_loader = TedataLoader(self.te_data_dir, self.transform_te,)
        
        self.tr_data_loader = DataLoader(self.tr_data_loader, batch_size=self.mini_batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.te_data_loader = DataLoader(self.te_data_loader, batch_size=self.mini_batch_size, shuffle=True, num_workers=4, drop_last=True)
        
        self.result_psnr_arr = []
        self.result_ssim_arr = []
        self.result_denoised_img_arr = []
        self.result_te_loss_arr = []
        self.result_tr_loss_arr = []
        self.best_psnr = 0
        
        self.save_file_name = _date+ '_N2N_' + _model_name + '_' + _training_type + '_noise_' + str(_noise_dist) + '_mbs_' + str(_mini_batch_size) + '_' + _case
        
            
        self._compile()
        
    def _compile(self):
        
        if self.model_name == 'DnCNN':
            self.model = DnCNN(channels=1, num_of_layers=17)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=self.epochs/4, factor=0.5, verbose=True)
        self.loss = nn.MSELoss()
        
        self.model = self.model.cuda()
        self.loss = self.loss.cuda()
        
    def get_PSNR(self, X, X_hat):
        
        mse = mean_squared_error(X,X_hat)
        test_PSNR = 10 * math.log10(1/mse)
        
        return test_PSNR
    
    def get_SSIM(self, X, X_hat):
        
        test_SSIM = measure.compare_ssim(X, X_hat, dynamic_range=X.max() - X.min())
        
        return test_SSIM
        
    def save_model(self, epoch):

        torch.save(self.model.state_dict(), './weights/'+self.save_file_name  +'_ep'+ str(epoch) + '.w')
        return
        
    def eval(self):
        """Evaluates denoiser on validation set."""

#         self.model.train(False)
        
        psnr_arr = []
        ssim_arr = []
        loss_arr = []
        denoised_img_arr = []

        with torch.no_grad():
        
            for batch_idx, (source, target) in enumerate(self.te_data_loader):

                source = source.cuda()
                target = target.cuda()

                # Denoise
                source_denoised = self.model(source)

                # Update loss
                loss = self.loss(source_denoised, target)

                source_denoised = np.clip(np.nan_to_num(source_denoised.cpu().numpy()),0,1)
                target = target.cpu().numpy()
                loss = loss.cpu().numpy()
                
                # Compute PSRN
                for i in range(self.mini_batch_size):
                    loss_arr.append(loss)
                    psnr_arr.append(self.get_PSNR(source_denoised[i,0,:,:], target[i,0,:,:]))
                    ssim_arr.append(self.get_SSIM(source_denoised[i,0,:,:], target[i,0,:,:]))
                    denoised_img_arr.append(source_denoised[i,0,:,:])
                
        mean_loss = np.mean(loss_arr)
        mean_psnr = np.mean(psnr_arr)
        mean_ssim = np.mean(ssim_arr)
        
        if self.best_psnr <= mean_psnr:
            self.best_psnr = mean_psnr
            self.result_denoised_img_arr = denoised_img_arr.copy()
            
        return mean_loss, mean_psnr, mean_ssim
        
    def _on_epoch_end(self,epoch,mean_tr_loss):
        """Tracks and saves starts after each epoch."""

        mean_te_loss, mean_psnr, mean_ssim = self.eval()
        
        self.result_psnr_arr.append(mean_psnr)
        self.result_ssim_arr.append(mean_ssim)
        self.result_te_loss_arr.append(mean_te_loss)
        self.result_tr_loss_arr.append(mean_tr_loss)

        print ('Tr loss : ', round(mean_tr_loss,4), ' Test loss : ', round(mean_te_loss,4), ' PSNR : ', round(mean_psnr,2), ' SSIM : ', round(mean_ssim,4))
        self.save_model(epoch+1)
        
#         self.save_model(epoch, stats, epoch == 0)
        
    def train(self):
        """Trains denoiser on training set."""

        num_batches = len(self.tr_data_loader)

        for epoch in range(self.epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.epochs))
            
            tr_loss = []

            for batch_idx, (source, target) in enumerate(self.tr_data_loader):

                source = source.cuda()
                target = target.cuda()
                
                # Denoise image
                source_denoised = self.model(source)
                loss = self.loss(source_denoised, target)

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                tr_loss.append(loss.detach().cpu().numpy())
                
            mean_tr_loss = np.mean(tr_loss)
            self._on_epoch_end(epoch, mean_tr_loss)            
        sio.savemat('./result_data/'+self.save_file_name,{'tr_loss_arr':self.result_tr_loss_arr, 'te_loss_arr':self.result_te_loss_arr, 
                                                          'psnr_arr':self.result_psnr_arr, 'ssim_arr':self.result_ssim_arr,
                                                         'denoised_img':self.result_denoised_img_arr,})

