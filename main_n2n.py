import torch
import torch.nn as nn
import torch.nn.functional as F

from core.noise2noise import Noise2Noise

#models for generator
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
#         model = [   nn.ReflectionPad2d(3),
#                     nn.Conv2d(input_nc, 64, 7),
#                     nn.InstanceNorm2d(64),
#                     nn.ReLU(inplace=True) ]
        
        self.pad2d_1 = nn.ReflectionPad2d(3)
        self.conv2d_1 = nn.Conv2d(input_nc, 64, 7)
        self.norm2d_1 = nn.InstanceNorm2d(64)
        self.relu_1 = nn.ReLU(inplace=True) 
        
        self.rv_embedding = nn.Linear(1, 64)
        self.tanh = nn.Tanh()
        
        model = []

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x, z):
        
        out1 = self.pad2d_1(x)
        out2 = self.conv2d_1(out1)
        
        
        out1_1 = self.rv_embedding(z)
        out1_2 = self.tanh(out1_1)
        out1_3 = out1_2.view(-1,64,1,1)

        out2_1 = out2*out1_3
        
        out3 = self.norm2d_1((out2_1))
        out4 = self.relu_1((out3))
        
        return self.model(out4)



_date = '190326'
_data_root_dir = './data/'
_weight_root_dir = './weights/' 
_train_name = 'gan_trdata_20500_patch_120x120.hdf5'
_test_name = 'BSD68_std25.mat'
_training_type = 'clean' # 'clean', 'noisy', 'GAN_single', 'GAN_averaged'
_g_weight_name = '190319_cycleGAN_wGAN_experiment_only_zi_zi-zi_hat_mbs16_not_equal_loss_x0001_gloss_x1_ep19.w'# required if _training_type == 'GAN'
_noise_dist = 25
_mini_batch_size = 32
_learning_rate =0.001
_model_name = 'DnCNN' #'Unet', 'DnCNN'
_crop_size = 100 #size for cropping
_avg_size = 4    #size for average (need for 'GAN_averaged')
_epochs = 100
_case = 'trdata_20500_tedata_BSD68'

_tr_data_dir = _data_root_dir + _train_name
_te_data_dir = _data_root_dir + _test_name

_g_weight_dir = _weight_root_dir + _g_weight_name

_generator = Generator(1,1)


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Initialize model and train
    n2n = Noise2Noise(_date, _tr_data_dir, _te_data_dir, _g_weight_dir, _training_type, _noise_dist, _epochs, _mini_batch_size, _learning_rate, _model_name, _case, _generator,_avg_size,_crop_size)
    n2n.train()









