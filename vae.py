import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import utils
from torch.utils.data import DataLoader



class VAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""

    def __init__(self, input_size,latent_size, hidden_size):
        super(VAE, self).__init__()

        # 定义均方差对象
        self.Loss_MSE = torch.nn.MSELoss()

        # 编码
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mean = nn.Linear( hidden_size, latent_size)
        self.fc_logvar = nn.Linear( hidden_size, latent_size)
        # 解码
        self.fc2= nn.Linear(latent_size, hidden_size)
        self.fc3 = nn.Linear( hidden_size, input_size)

    # 编码
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    # 解码
    def decode(self, z):
        h = F.relu(self.fc2(z))
        recon_x = torch.relu(self.fc3(h))
        return recon_x
    
    # 重参数采样
    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparametrize(mean, logvar)
        recon_x = self.decode(z)

        return recon_x, mean, logvar 

    def loss_function(self, recon_x, x, mean, logvar) -> torch.Tensor:
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*logvar- mean.pow(2) - torch.exp(2*logvar))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss
           
        


if __name__ == "__main__":
    epochs = 100
    recon = None
    img = None

    utils.make_dir("./img/vae")
    utils.make_dir("./model_weights/vae")

    train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    vae = VAE(input_size=784, latent_size=10, hidden_size=200)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    for epoch in range(100):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            img, label = data
            inputs = img.reshape(img.shape[0], -1)
            recon, mean,logvar = vae.forward(inputs)
            loss = vae.loss_function(recon, inputs, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            i += 1

            if batch_id % 100 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch+1, epochs, batch_id+1, len(data_loader), loss.item()))

        print("======>epoch:{},\t epoch_average_batch_loss:{:.6f}============".format(
            epoch+1, train_loss/i), "\n")

        # save imgs
        if epoch % 10 == 0:
            # 查看图像
            imgs = utils.to_img(recon.detach())
            path = "./img/vae/epoch{}.png".format(epoch+1)
            torchvision.utils.save_image(imgs, path, nrow=10)
            print("save:", path, "\n")

    torchvision.utils.save_image(img, "./img/cvae/raw.png", nrow=10)
    print("save raw image:./img/vae/raw/png", "\n")

    # save val model
    utils.save_model(vae, "./model_weights/vae/vae_weights.pth")