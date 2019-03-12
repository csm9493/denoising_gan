from core.trainer import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        
input_nc = 1
output_nc = 1

lr_g = 2e-4
lr_c = 5e-5
ep = 50
decay_ep = 25
mbs = 8
im_size = 100
tr_data_name = 'gan_trdata_20500_patch_120x120.hdf5'
critic_iter = 5
gpu_num = 2
tr_type = 'only_zi'

experiment_type = ''
sv_name = '190313_cycleGAN_wGAN_experiment_'+ tr_type + experiment_type

netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)
netD_B = Discriminator(input_nc)


netG_A2B.cuda()
netG_B2A.cuda()
netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

trainer = Trainer(netG_A2B, netG_B2A, netD_B, save_name=sv_name, lr_g=lr_g, lr_critic=lr_c, epochs=ep, decay_epoch = decay_ep, 
                  mini_batch_size=mbs, img_size=im_size, tr_data_name = tr_data_name, critic_iter = critic_iter,
                  gpu_num = gpu_num, input_nc=input_nc, output_nc = output_nc, train_type=tr_type)



