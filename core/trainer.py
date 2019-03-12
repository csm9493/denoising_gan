import itertools
import torchvision.transforms as transforms
from .utils import NoisyImageDataset
from .utils import Logger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class Trainer():
    
    def __init__(self, netG_A2B, netG_B2A, netD_B, lr_g=2e-4, lr_critic=5e-5, epochs=50, decay_epoch = 25, mini_batch_size=8, img_size=100, tr_data_name = None, critic_iter = 5, gpu_num = 0, input_nc=1, output_nc = 1):
        
        self.netG_A2B = netG_A2B
        self.netG_B2A = netG_B2A
        self.netD_B = netD_B
        
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        
        self.n_epochs = epochs
        self.critic_iter = critic_iter
        
        # Optimizers & LR schedulers
        self.optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.RMSprop(netD_B.parameters(), lr=lr_critic)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)

        # Dataset loader
        transforms_ = [ transforms.RandomCrop(img_size), 
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)) ]
        
        transformed_dataset = NoisyImageDataset(hdf5_file=tr_data_name, root_dir='data/', transform=transforms.Compose(transforms_))
        self.dataloader = DataLoader(transformed_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=4, drop_last=True)

        self.logger = Logger(self.n_epochs, len(self.dataloader))

        cuda_index = gpu_num
        
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor
        self.input_A = Tensor(mini_batch_size, input_nc, img_size, img_size)
        
        self.train()
        
    def train(self):
        
        for epoch in range(self.n_epochs):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch))

                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update

                for d_iter in range(self.critic_iter):
                    # Train discriminator
                    self.optimizer_D_B.zero_grad()

                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.netD_B(real_A)
                    d_loss_real = d_loss_real.mean()
        #             d_loss_real.backward(mone)

                    # Train with fake images
                    fake_B = self.netG_A2B(real_A).detach()
                    d_loss_fake = self.netD_B(fake_B)
                    d_loss_fake = d_loss_fake.mean()
        #             d_loss_fake.backward(one)

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
                fake_B = self.netG_A2B(real_A)
                g_loss = self.netD_B(fake_B)
                g_loss = -g_loss.mean()*1
                g_cost = g_loss

                # Identity loss
                # G_B2A(A) should equal A if real A is fed
                same_A = self.netG_B2A(real_A)
                loss_identity_A = self.criterion_identity(same_A, real_A)*5.0

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*10.0

                # Total loss
                loss_G = loss_identity_A  + g_loss + loss_cycle_ABA
                loss_G.backward()
                self.optimizer_G.step()

                # Progress report (http://localhost:8085)
                self.logger.log({'loss_G': loss_G, 'loss_identity_A': (loss_identity_A), 'loss_cycle_ABA': (loss_cycle_ABA),'g_cost': (g_cost), 'Wasserstein_D': (Wasserstein_D)}, 
                            images={'Zi': real_A[0], 'Zj_hat': fake_B[0], 'Zi_tilde': recovered_A[0]})