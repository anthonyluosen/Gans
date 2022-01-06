import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import  torchvision.transforms  as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
class Discrimnator(nn.Module):
    def __init__(self,input_features):
        super(Discrimnator,self).__init__()
        self.input_dim = input_features
        self.disc = nn.Sequential(
            nn.Linear(input_features,256),
            nn.LeakyReLU(0.01),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x ):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self,noise_dim,img_dim):
        super(Generator, self).__init__()
        self.Gen = nn.Sequential(
            nn.Linear(noise_dim,256),
            nn.LeakyReLU(0.01),
            nn.Linear(256,img_dim),
            nn.Tanh()
        )
    def forward(self,x):
        return self.Gen(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
z_dim  = 64
img_dim = 28*28
batch_size = 32
num_epochs = 40

disc = Discrimnator(img_dim).to(device)
gen = Generator(noise_dim=z_dim, img_dim=img_dim).to(device)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),]
)
fixed_noise = torch.randn((batch_size,z_dim)).to(device)
dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

step = 0

for epoch in range(num_epochs):
    for idx,(real,target) in enumerate(loader):
        real = real.view(-1,784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn((batch_size,z_dim)).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_d = (lossD_fake+lossD_real)/2
        disc.zero_grad()
        loss_d.backward(retain_graph = True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        out_put = disc(fake).view(-1)
        lossG = criterion(out_put,torch.ones_like(out_put))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {idx}/{len(loader)} \
                              Loss D: {loss_d:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
