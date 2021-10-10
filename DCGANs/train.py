#-*-coding =utf-8 -*-
#@time :2021/10/10 19:00
#@Author: Anthony
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from Gans.model import Generator,Discriminator,initialize_weight

device = ('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 2e-4
num_epochs = 5
features_g = 64
features_d = 64
batch_size = 124
channels_img = 1
noise_dim = 100
img_size = 64

transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)],
        )
    ]
)

data = datasets.MNIST(root='datasets/', download=True,transform=transform,train=True)
loader = DataLoader(dataset=data,batch_size=batch_size,shuffle=True)

gen = Generator(noise_dim=noise_dim,features_g=features_g,channels_img=channels_img).to(device)
disc = Discriminator(channels_img=channels_img,features_d=features_d).to(device)

initialize_weight(gen)
initialize_weight(disc)

opt_gen = optim.Adam(gen.parameters(),lr=learning_rate,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=learning_rate,betas=(0.5,0.999))

criterion = torch.nn.BCELoss()
fixed_noise = torch.randn((32,noise_dim,1,1)).to(device)

writer_real = SummaryWriter(f'logs/real')
writer_fake = SummaryWriter(f'logs/fake')
step = 0

disc.train()
gen.train()

for epoch in range(num_epochs):
    for batch_idx,(real,_) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(batch_size,noise_dim,1,1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_f = criterion(disc_fake,torch.zeros_like(disc_fake))
        disc_real = disc(real).reshape(-1)
        loss_r = criterion(disc_real,torch.ones_like(disc_real))
        loss_d = (loss_f+loss_r)/2

        disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {loss_d:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32],normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                writer_fake.add_image('real',img_grid_fake,global_step=step)
                writer_real.add_image('real',img_grid_real,global_step=step)

            step +=1









