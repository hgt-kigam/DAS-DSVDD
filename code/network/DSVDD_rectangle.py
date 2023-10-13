import torch.nn as nn
import torch
import easydict
from torchsummary import summary

class Deep_SVDD(nn.Module):
    def __init__(self, args):
        super(Deep_SVDD, self).__init__()
        self.num_filter = args.num_filter
        self.latent_dim = args.latent_dim

        self.conv_1 = conv_block(1, self.num_filter)
        self.conv_2 = conv_block(self.num_filter * 1, self.num_filter * 2)
        self.conv_3 = conv_block(self.num_filter * 2, self.num_filter * 4)
        self.conv_4 = conv_block(self.num_filter * 4, self.num_filter * 8)
        self.linear_1 = nn.Linear(
            self.num_filter * 8 * 13 * 6, self.latent_dim, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        encoded = self.linear_1(x)
        return encoded


class pretrain_autoencoder(nn.Module):
    def __init__(self, args):
        super(pretrain_autoencoder, self).__init__()
        self.num_filter = args.num_filter
        self.latent_dim = args.latent_dim

        self.conv_1 = conv_block(1, self.num_filter)
        self.conv_2 = conv_block(self.num_filter * 1, self.num_filter * 2)
        self.conv_3 = conv_block(self.num_filter * 2, self.num_filter * 4)
        self.conv_4 = conv_block(self.num_filter * 4, self.num_filter * 8)

        self.linear_1 = nn.Linear(
            self.num_filter * 8 * 13 * 6, self.latent_dim, bias=False)
        self.linear_2 = nn.Linear(
            self.latent_dim, self.num_filter * 8 * 13 * 6, bias=False)

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (self.num_filter * 8, 13, 6))

        self.trans_1 = conv_trans_block_1(
            self.num_filter * 8, self.num_filter * 4)
        self.trans_2 = conv_trans_block(
            self.num_filter * 4, self.num_filter * 2)
        self.trans_3 = conv_trans_block(
            self.num_filter * 2, self.num_filter * 1)
        self.trans_4 = conv_trans_block(self.num_filter * 1, 1)

    def encoder(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        encoded = self.linear_1(x)
        return encoded
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        encoded = self.linear_1(x)
        x = self.linear_2(encoded)
        x = self.unflatten(x)
        x = self.trans_1(x)
        x = self.trans_2(x)
        x = self.trans_3(x)
        decoded = self.trans_4(x)
        return decoded


def conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=3, padding=3, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_dim, eps=1e-04, affine=False),
        nn.Conv2d(out_dim, out_dim, kernel_size=7, padding='same', bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_dim, eps=1e-04, affine=False)
    )
    return model


def conv_trans_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=7, stride=3, padding=3 , bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_dim, eps=1e-04, affine=False),
        nn.Conv2d(out_dim, out_dim, kernel_size=7, padding='same', bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_dim, eps=1e-04, affine=False)
    )
    return model

def conv_trans_block_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=7, stride=3, padding=3, output_padding=(1,0), bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_dim, eps=1e-04, affine=False),
        nn.Conv2d(out_dim, out_dim, kernel_size=7, padding='same', bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_dim, eps=1e-04, affine=False)
    )
    return model

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# args = easydict.EasyDict({
#         'num_filter':32,
#         'latent_dim':100
#                 })
# ae = Deep_SVDD(args).to(device)
# print(summary(ae, (1, 1000, 406)))