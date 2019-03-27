from visdom import Visdom
import sys
import random
import time
import datetime
import numpy as np
import scipy.io as sio

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvF
import h5py

class NoisyImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, hdf5_file, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.data = h5py.File(root_dir+hdf5_file, "r")
        self.num_data = self.data["noisy"].shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        
        img = Image.fromarray((self.data["noisy"][idx,:,:]/255.))
        sample = img

        if self.transform:
            sample = self.transform(sample)

        return sample

def tensor2image(tensor):
#     image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    image = np.clip(0.5*(tensor[0].cpu().float().numpy() + 1.0)*255, 0, 255)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, sv_file_name):
        self.viz = Visdom(port='8085')
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.save_file_name = sv_file_name
        self.loss_save = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data.cpu().numpy()
            else:
                self.losses[loss_name] += losses[loss_name].data.cpu().numpy()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                    self.loss_save[loss_name] = []
                    self.loss_save[loss_name].append(loss/self.batch)
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                    self.loss_save[loss_name].append(loss/self.batch)
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            sio.savemat(self.save_file_name,{loss_name:self.loss_save[loss_name] for loss_name in self.loss_save})
                        
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1
            
class TrdataLoader():

    def __init__(self,_tr_data_dir=None, _training_type='clean', _g_weight_dir = None, _noise_dist=25, _transform=None, _generator = None, _avg_size = 4):

        self.tr_data_dir = _tr_data_dir
        self.training_type = _training_type
        self.g_weight_dir = _g_weight_dir
        self.noise_dist = _noise_dist
        self.transform = _transform
        self.G = _generator
        self.avg_size = _avg_size
        
        self.data = h5py.File(self.tr_data_dir, "r")
        self.num_data = self.data["clean"].shape[0]
        
        if self.training_type == 'GAN_single' or self.training_type == 'GAN_averaged':
            self.G.load_state_dict(torch.load(self.g_weight_dir))
        
        print ("training type : ", self.training_type)
            
        
    def __len__(self):
        return self.num_data


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        size = img.shape[1]
        c = img.shape[0]

        noise = np.random.normal(0, self.noise_dist/255., (c, size, size))

        # Add noise and clip
        noise_img = np.array(img).reshape(c,size,size) + noise

#         noise_img = np.clip(noise_img, 0, 1).astype(np.uint8)
        return noise_img.astype(np.float32)

    def _corrupt(self, img):

        return self._add_noise(img)


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        if self.training_type == 'clean' or self.training_type == 'noisy':
            img = Image.fromarray((self.data["clean"][index,:,:]/255.))

            if self.transform:
                img = self.transform(img)

            # Corrupt source image
            tmp = self._corrupt(img)
            source = (self._corrupt(img) - 0.5) / 0.5

            # Corrupt target image, but not when clean targets are requested
            if self.training_type=='clean':
                target = img
            else:
                target = tmp
        
        else:
            img = Image.fromarray((self.data["noisy"][index,:,:]/255.))
#             Tensor = torch.cuda.FloatTensor
           
            

            if self.transform:
                img = self.transform(img)
                size = img.shape[2]
                
            with torch.no_grad():
                if self.training_type == 'GAN_single':
                    noise = torch.Tensor(1,1,1,1).normal_(0, 1)
                    noisev = Variable(noise)

                    source = img.view(1,1,size,size)
                    target = self.G(source, noisev)

                    target = (target*0.5) + 0.5 # change the range of generated noisy image to [0,1]
                    source = source.view(1,size,size) # reshape for training
                    target = source.view(1,size,size)
                    
                else:
                    
                    copied_inputs = torch.Tensor(self.avg_size, 1, size, size)
                    
                    noise = torch.Tensor(self.avg_size,1,1,1).normal_(0, 1)
                    noisev = Variable(noise)

                    copied_sources = copied_inputs.copy_(img[0])
                    
                    targets = self.G(copied_sources, noisev)

                    targets = (targets*0.5) + 0.5 # change the range of generated noisy image to [0,1]
                    source = img.view(1,size,size) # reshape for training
                    target = torch.mean(targets,0).view(1,size,size)

        return source, target


class TedataLoader():

    def __init__(self,_tedata_dir=None, _transform=None):

        self.te_data_dir = _tedata_dir
        self.transform = _transform

        self.data = sio.loadmat(self.te_data_dir)
        self.num_data = self.data["clean_images"].shape[0]
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        source = Image.fromarray((self.data["noisy_images"][index,:,:]/255.))
        target = Image.fromarray((self.data["clean_images"][index,:,:]/255.))
        
        if self.transform:
            source = self.transform(source)
            
#         source = tvF.to_tensor(source)
        target = tvF.to_tensor(target)

        return source, target