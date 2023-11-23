import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import utils
from torch.utils.data import DataLoader


'''
这是第二版HCVAE，第一版的问题是，loss会变成nan值
第二版改变了计算loss函数的方式，不用HVAE的均值和方差计算
而是使用传统的kl散度和重构误差之和作为loss函数值看看效果
'''
class HCVAE2(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""

    def __init__(self, feature_size, class_size, latent_size):
        super(HCVAE2, self).__init__()

        # 定义均方差对象
        self.Loss_MSE = torch.nn.MSELoss()

        # 定义网络
        self.fc2_mu = nn.Linear(200, latent_size)
        self.fc2_log_std = nn.Linear(200, latent_size)
        self.fc1_mu = nn.Linear(200, feature_size)
        self.fc1_log_std = nn.Linear(200, feature_size)
        # 编码
        self.encoder_fc1 = nn.Linear(feature_size + class_size, 200)
        self.encoder_fc2 = nn.Linear(feature_size + class_size, 200)
        self.encoder_fc3 = nn.Linear(feature_size + class_size, 200)

        # 解码
        self.decoder_fc1 = nn.Linear(latent_size + class_size, 200)
        self.decoder_fc2 = nn.Linear(feature_size + class_size, 200)
        self.decoder_fc3 = nn.Linear(feature_size + class_size, 200)
        self.decoder_mu = nn.Linear(200, feature_size)
        self.decoder_log_std = nn.Linear(200, feature_size)

    def encode1_2(self, func, x, y):
        # concat features and labels
        h1 = F.relu(func(torch.cat([x, y], dim=1)))
        mu = self.fc1_mu(h1)
        log_std = self.fc1_log_std(h1)
        return mu, log_std

    def encode3(self, x, y):
        h1 = F.relu(self.encoder_fc3(torch.cat([x, y], dim=1)))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode1(self, z, y):
        # concat latents and labels
        h3 = F.relu(self.decoder_fc1(torch.cat([z, y], dim=1)))
        # 这里decoder也是先decoder出来均值和方差，因为，后面计算loss函数要用
        # 在decoder后再使用reparametrize重采样出来一个z放入下一层解码
        de_mu = self.fc1_mu(h3)
        de_log_std = self.fc1_log_std(h3)

        return de_mu, de_log_std
        

    def decode2_3(self, z, y):
        # concat latents and labels
        h3 = F.relu(self.decoder_fc3(torch.cat([z, y], dim=1)))
        # 这里decoder也是先decoder出来均值和方差，因为，后面计算loss函数要用
        # 在decoder后再使用reparametrize重采样出来一个z放入下一层解码
        de_mu = self.fc1_mu(h3)
        de_log_std = self.fc1_log_std(h3)
        return de_mu, de_log_std

    def final_decode(self, z, y):
        h3 = F.relu(self.decoder_fc3(torch.cat([z, y], dim=1)))
        mu = self.decoder_mu(h3)
        log_std = self.decoder_log_std(h3)
        return mu, log_std

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, y):
        # 第一次条件编码
        mu_1, log_std_1 = self.encode1_2(self.encoder_fc1, x, y)
        z_1 = self.reparametrize(mu_1, log_std_1)

        # 第二次条件编码
        mu_2, log_std_2 = self.encode1_2(self.encoder_fc2, z_1, y)
        z2 = self.reparametrize(mu_2, log_std_2)

        # 第三次条件编码
        mu_3, log_std_3 = self.encode3(z2, y)
        z3 = self.reparametrize(mu_3, log_std_3)

        # 第一次条件解码
        # 先解码出重构值，再根据重构值计算其均值和方差
        de_mu3, de_log3 = self.decode1(z3, y)
        recon3 = self.reparametrize(de_mu3, de_log3)

        # 第二次条件解码
        de_mu2, de_log2 = self.decode2_3(recon3, y)
        recon2 = self.reparametrize(de_mu2, de_log2)

        # 第三次条件解码
        de_mu1, de_log1 = self.final_decode(recon2, y)
        recon1 = self.reparametrize(de_mu1, de_log1)

        # 根据计算loss函数所用到的内容
        # 将编码和解码得到的均值打包成方差
        en_mu = [mu_1, mu_2, mu_3]
        de_mu = [de_mu3, de_mu2, de_mu1]
        log_std = [log_std_1, log_std_2, log_std_3]
        recon = [recon1, recon2, recon3]
        z = [z_1, z2, z3]

        #loss = self.loss_function(recon, x,  en_mu, de_mu, log_std)
        return recon, z, en_mu, de_mu, log_std

    def loss_function(self, recon, z, x, en_mu, de_mu, log_std) -> torch.Tensor:
        # 计算重构误差
        recon_loss = F.mse_loss(recon[0], x, reduction="sum") 
        recon_loss = recon_loss + F.mse_loss(recon[1], z[0], reduction="sum" )
        recon_loss = recon_loss + F.mse_loss(recon[2], z[1], reduction="sum" )
        # 计算KL散度
        kl_loss = torch.pow((en_mu[0] - de_mu[1]), 2)
        kl_loss = kl_loss +  torch.pow((en_mu[1] - de_mu[0]), 2)
        sum_log = torch.sum(log_std[0])+torch.sum(log_std[1])+torch.sum(log_std[2])
        kl_loss = torch.sum(kl_loss) - sum_log
        
        # 计算整体的loss函数
        loss =  kl_loss +recon_loss
        return loss


if __name__ == "__main__":
    epochs = 100
    batch_size = 100

    recon = None
    img = None

    utils.make_dir("./img/cvae")
    utils.make_dir("./model_weights/cvae")

    train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    cvae = HCVAE2(feature_size=784, class_size=10, latent_size=10)
    #cvae = CVAE(feature_size=784, class_size=10, latent_size=10)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-4)

    for epoch in range(100):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            img, label = data
            inputs = img.reshape(img.shape[0], -1)
            y = utils.to_one_hot(label.reshape(-1, 1), num_class=10)
            #recon, mu, log_std = cvae(inputs, y)
            #loss = cvae.loss_function(recon, inputs, mu, log_std)
            recon, z,en_mu, de_mu, log_std = cvae.forward(inputs, y)
            print(recon.shape)
            loss = cvae.loss_function(recon, z, inputs, en_mu, de_mu, log_std)
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
            imgs = utils.to_img(recon[0].detach())
            path = "./img/hcvae/hepoch{}.png".format(epoch+1)
            torchvision.utils.save_image(imgs, path, nrow=10)
            print("save:", path, "\n")

    torchvision.utils.save_image(img, "./img/cvae/raw.png", nrow=10)
    print("save raw image:./img/cvae/raw/png", "\n")

    # save val model
    utils.save_model(cvae, "./model_weights/cvae/cvae_weights.pth")