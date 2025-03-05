# Do not add any other packages
# Use these packages to write your code
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

# Transforming celeba dataset
def transform_fn(is_training):
    if is_training:
        return T.Compose([
            T.Resize(70),
            T.RandomCrop(64),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    elif not is_training:
        return T.Compose([
            T.Resize(70),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])



# Celeba dataset class
# We use 10,000 images as train set and 1,000 images as evaluation set
class CelebADataset(Dataset):

    def __init__(self, args, is_train):
        super().__init__()
        self.transform = transform_fn(is_train)
        img_list = sorted(os.listdir(args.img_path))

        if is_train is True:
            img_list = img_list[:10000]
        elif is_train is False:
            img_list = img_list[10000:]
        self.img_list = img_list
        self.img_path = args.img_path

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        image = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')
        image = self.transform(image)
        return image

def generate_noise(batch_size, noise_size, seed):
    seed = seed + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return torch.randn(batch_size, noise_size) , seed


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--img_path', type=str, default = "./celeba")
    parser.add_argument('--stop_epoch', type=int, default = 10)
    parser.add_argument('--output_path', type=str, default = ".")
    args = parser.parse_args()

    assert args.img_path is not None, 'image path should be specified'
    assert args.output_path is not None, 'output path should be specified'

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    # Dataloader for celeba dataset
    # Use "train_dataloader" and "eval_dataloader" for in your code

    train_dataset = CelebADataset(args, is_train=True)
    eval_dataset = CelebADataset(args, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=2)

    class Generator(nn.Module):
      def __init__(self):
          super(Generator, self).__init__()
          self.main = nn.Sequential(
              nn.Linear(100, 256),
              nn.LeakyReLU(0.1),
              nn.Linear(256, 512),
              nn.LeakyReLU(0.1),
              nn.Linear(512, 64),
              nn.LeakyReLU(0.1),
              nn.Linear(64, 64),
              nn.LeakyReLU(0.1),
              nn.Linear(64, 64 * 64 * 3),
              nn.Sigmoid(),
          )
      def forward(self, input):
          output = self.main(input)
          # output shape = batch_size, 3, 64, 64
          output = output.view(-1, 3, 64, 64)
          return output

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                # Output size = ((Input size - Kernel size + 2 * Padding) / Stride) + 1
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
                # 32 * 32
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=2),
                # 16 * 16 
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
                # 8 * 8
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                # 4 * 4 
                nn.ReLU(),
                # 1 = ((4 - k +2 * 0)/1) + 1 => k= 4
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0), 
                nn.Sigmoid()
            )
        def forward(self, input):
            output = self.main(input)
            return output.view(-1, 1).squeeze(1)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Generator().cuda()
    netD = Discriminator().cuda()
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = generate_noise(32, 100, seed)

    # Establish convention for real and fake labels during training


    optimizerG = optim.AdamW(netG.parameters(), lr=0.0001)
    optimizerD = optim.AdamW(netD.parameters(), lr=0.0001)


    criterion = nn.BCELoss()
    G_losses = []
    D_losses = []
    iters = 0
    total_loss=0
    
    # the total epoch is set to 5

    for epoch in range(5):
        for batch_idx, data in enumerate(train_dataloader):
    
            netD.zero_grad()
            netG.zero_grad() 
            data = data.cuda()
            batch_size = data.size(0)
            real_label = torch.ones((batch_size,)).cuda()
            output = netD(data)
            errD_real = criterion(output, real_label)
            D_x = output.mean().item()

            # train with fake
            noise = generate_noise(batch_size, 100,seed)[0].cuda()
            fake = netG(noise)
            fake_label = torch.zeros((batch_size,)).cuda()
            output = netD(fake.detach())
            errD_fake = criterion(output, fake_label)
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            netG.zero_grad()
            label = torch.ones((batch_size,)).cuda() 
            # label should be 1 fake labels are real for generator cost
            # because it G wants to fool D
            output = netD(fake)
            errG = criterion(output, label)
            D_G_z2 = output.mean().item()

            errG.backward()
            optimizerG.step()
            total_loss += errD.item() + errG.item()

        eval_loss_t = 0.0
        for batch_idx, eval_data in enumerate(eval_dataloader):
            batch_size = eval_data.size(0)
            eval_data = eval_data.cuda()
            output_real = netD(eval_data)
            label_real = torch.ones((batch_size,)).cuda()
            label_fake = torch.zeros((batch_size,)).cuda()
        
            noise = generate_noise(batch_size, 100,seed)[0].cuda()
            fake = netG(noise)
            output_fake = netD(fake)
            # errG = criterion(output_fake, label)


            D_loss_real = criterion(output_real, label_real)
            D_loss_fake = criterion(output_fake, label_fake)
            D_loss = D_loss_real + D_loss_fake

            G_loss= criterion (output_fake,label_real)
           
            eval_loss_t += D_loss + G_loss

        eval_loss=eval_loss_t.item()


      # At argument's stop epoch, below code writes your evaluation loss (at certain epoch) to txt file
      # Then, it stops the for loop
        if epoch == args.stop_epoch:
            loss = round(eval_loss, 3)
            print(loss)

            script_path = __file__
            script_name = os.path.basename(script_path)
            script_name = script_name[script_name.rfind("/")+1:]
            script_name = script_name[:script_name.rfind(".")]
            with open(os.path.join(args.output_path, 'result.txt'), 'a') as f:
                f.write(f"{script_name}\t{str(loss)}\n")
            break

if __name__ == '__main__':
    main()
