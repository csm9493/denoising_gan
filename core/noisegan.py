import itertools
import torchvision.transforms as transforms
from .utils import NoisyImageDataset
from .utils import Logger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

def custom_loss(output, target):
    loss = torch.log(torch.mean((output - target)**2))
    return -loss

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class NoiseGAN():
    
    def __init__(self, netG_A2B, netG_B2A, netD_B, save_name=None, lr_g=2e-4, lr_critic=5e-5, epochs=50, decay_epoch = 25, mini_batch_size=8, img_size=100, tr_data_name = None, critic_iter = 5, input_nc=1, output_nc = 1, train_type = 'shuffle', a_gloss = 1.0, a_not_equal_loss = 0.0001, a_cycle_loss = 10):
        
        self.netG_A2B = netG_A2B
        self.netG_B2A = netG_B2A
        self.netD_B = netD_B
        
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.criterion_not_equal = custom_loss
        
        self.save_weight_name = './weights/' + save_name + '_ep'
        self.save_file_name = './result_data/' + save_name + '_result.mat'
        self.n_epochs = epochs
        self.critic_iter = critic_iter
        self.mini_batch_size = mini_batch_size
        self.train_type = train_type
        self.a_gloss = a_gloss
        self.a_not_equal_loss = a_not_equal_loss
        self.a_cycle_loss = a_cycle_loss
        
        # Optimizers & LR schedulers
        self.optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.RMSprop(netD_B.parameters(), lr=lr_critic)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)

        # Dataset loader
        transforms_ = [ transforms.RandomCrop(img_size), 
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)) ]
        
        transformed_dataset = NoisyImageDataset(hdf5_file=tr_data_name, root_dir='data/', transform=transforms.Compose(transforms_))
        self.dataloader = DataLoader(transformed_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=4, drop_last=True)

        self.logger = Logger(self.n_epochs, len(self.dataloader), self.save_file_name)
        
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor
        self.input_A = Tensor(mini_batch_size, input_nc, img_size, img_size)
        self.noise = torch.Tensor(mini_batch_size, 1, 1, 1)
        
        self.train()
        
    def train(self):
        
        for epoch in range(self.n_epochs):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                if self.train_type == 'shuffle':
                    real_A = Variable(self.input_A.copy_(batch))
                else:
                    real_A = Variable(self.input_A.copy_(batch[0]))
                
                self.noise.resize_(self.mini_batch_size, 1, 1, 1).normal_(0, 1)
                noisev = Variable(self.noise).cuda()

                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update

                for d_iter in range(self.critic_iter):
                    # Train discriminator
                    self.optimizer_D_B.zero_grad()

                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.netD_B(real_A)
                    d_loss_real = d_loss_real.mean()

                    # Train with fake images
                    fake_B = self.netG_A2B(real_A, noisev)
                    d_loss_fake = self.netD_B(fake_B)
                    d_loss_fake = d_loss_fake.mean()

                    d_loss = -(d_loss_real -d_loss_fake)
                    Wasserstein_D = d_loss_real - d_loss_fake
                    d_loss.backward()
                    self.optimizer_D_B.step()

                    for p in self.netD_B.parameters():
                        p.data.clamp_(-0.01, 0.01)
                ###################################
                
                ###### Generators A2B and B2A ######
                self.optimizer_G.zero_grad()

                # GAN loss
                fake_B = self.netG_A2B(real_A, noisev)
                g_loss = self.netD_B(fake_B)
                g_loss = -g_loss.mean()*self.a_gloss
                g_cost = g_loss

                # not equal loss
                loss_not_equal = self.criterion_not_equal(fake_B, real_A)*self.a_not_equal_loss

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B, noisev)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*self.a_cycle_loss

                # Total loss
                loss_G = loss_not_equal  + g_loss + loss_cycle_ABA
                loss_G.backward()
                self.optimizer_G.step()
                
                zi_zi_hat_mean = ((real_A*0.5 + 0.5)-(fake_B*0.5 + 0.5)).mean()
                zi_zi_hat_var = ((real_A*0.5 + 0.5)-(fake_B*0.5 + 0.5)).var()

                # Progress report (http://localhost:8085)
                self.logger.log({'loss_G': loss_G, 'loss_not_equal': (loss_not_equal), 'loss_cycle_ABA': (loss_cycle_ABA), 'g_cost': (g_cost), 'Wasserstein_D': (Wasserstein_D), 'zi_zi_hat_mean': (zi_zi_hat_mean*255), 'zi_zi_hat_var': (zi_zi_hat_var*255)},images={'Zi': real_A[0], 'Zj_hat': fake_B[0], 'Zi_tilde': recovered_A[0], 'Zi-Zi_hat': real_A[0]-fake_B[0]})
                
            #save generator netG_A2B
            torch.save(self.netG_A2B.state_dict(), self.save_weight_name + str(epoch) + '.w')