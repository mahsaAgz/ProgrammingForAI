# Do not add any other packages
# Use these packages to write your code
# Do not change

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
# Do not change

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
# Do not change

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


# Function that produces noise vector
# The output is noise_vector and seed which is added with one
# Use like below
# noise, seed = generate_noise(batch_size, noise_size, seed)
# You must use your this code when defining your noise vector
# Otherwise, the result will be different from TA's
# Do not change

def generate_noise(batch_size, noise_size, seed):
    seed = seed + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return torch.randn(batch_size, noise_size) , seed



def main():

    # This code takes "seed", "image path" "epoch to stop" and "output path for the result" for argument
    # Do not change

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--img_path', type=str, default = "./celeba")
    parser.add_argument('--stop_epoch', type=int, default = 10)
    parser.add_argument('--output_path', type=str, default = ".")
    args = parser.parse_args()

    assert args.img_path is not None, 'image path should be specified'
    assert args.output_path is not None, 'output path should be specified'


    # Fix the seed for reproducibilty
    # Do not change

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


    ################################################################
   
    class Generator(nn.Module):
      def __init__(self):
          super(Generator, self).__init__()
          self.main = nn.Sequential(
              nn.Linear(100, 256),
              nn.LeakyReLU(0.2),
              nn.Linear(256, 64),
              nn.LeakyReLU(0.2),
              nn.Linear(64, 64),
              nn.LeakyReLU(0.2),
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
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    optimizerG = optim.AdamW(netG.parameters(), lr=0.0001)
    optimizerD = optim.AdamW(netD.parameters(), lr=0.0001)

#############################
    # fixed_noise= generate_noise(batch_size, 100,seed)[0].to(device)
    criterion = nn.BCELoss()
    total_loss=0
    ################################################################

    # the total epoch is set to 5
    # do not change
    for epoch in range(5):
        for batch_idx, data in enumerate(train_dataloader):
            ###################################################################
            ## TODO : Train and evaluate your model
            ## change below XXX into your loss variable on whole evaluation set
            ####################################################################

            ####################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
            ###################################################
            # train with real
            netD.zero_grad()
            netG.zero_grad() 
            data = data.to(device)
            batch_size = data.size(0)
            label = torch.ones((batch_size,)).to(device)
            output = netD(data)
            errD_real = criterion(output, label)
            D_x = output.mean().item()

            # train with fake
            noise = generate_noise(batch_size, 100,seed)[0].to(device)
            fake = netG(noise)
            label = torch.zeros((batch_size,)).to(device)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            ########################################
            # (2) Update G network: maximize log(D(G(z))) #
            ########################################
            netG.zero_grad()
            label = torch.ones((batch_size,)).to(device) 
            # label should be 1 fake labels are real for generator cost
            # because it G wants to fool D
            output = netD(fake)
            errG = criterion(output, label)
            D_G_z2 = output.mean().item()

            errG.backward()
            optimizerG.step()
        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, n_epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        #save the output
        # fake = netG(fixed_noise)
        # training_progress_images_list = save_gif(training_progress_images_list, fake)  # Save fake image while training!

        # # Check pointing for every epoch
        # torch.save(netG.state_dict(), './checkpoint/netG_epoch_%d.pth' % (epoch))
        # torch.save(netD.state_dict(), './checkpoint/netD_epoch_%d.pth' % (epoch))

        total_loss += errD.item() + errG.item()

        eval_loss = 0.0
        for batch_idx, eval_data in enumerate(eval_dataloader):
            batch_size = eval_data.size(0)
            eval_data = eval_data.to(device)
            output_real = netD(eval_data)
            label_real = torch.ones((batch_size,)).to(device)
            D_loss=criterion (output_real,label_real)

            noise = generate_noise(batch_size, 100,seed)[0].to(device)
            fake = netG(noise)
            output_fake = netD(fake)
            label_fake = torch.zeros((batch_size,)).to(device)
            # errG = criterion(output_fake, label)
            G_loss= criterion (output_fake,label_fake)


            eval_loss += D_loss + G_loss


      # At argument's stop epoch, below code writes your evaluation loss (at certain epoch) to txt file
      # Then, it stops the for loop
      # Do not change
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

