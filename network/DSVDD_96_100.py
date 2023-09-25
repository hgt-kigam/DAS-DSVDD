import torch.nn as nn


class Deep_SVDD_96_100(nn.Module):
    def __init__(self, args):
        super(Deep_SVDD_96_100, self).__init__()
        self.args = args
        start=16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, start, 5, bias=False, padding=2),          # 1*96*96->16*96*96
            nn.BatchNorm2d(start, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 16*96*96->16*48*48
            nn.Conv2d(start, start*2, 5, bias=False, padding=2),    # 16*48*48->32*48*48
            nn.BatchNorm2d(start*2, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 32*48*48->32*24*24
            nn.Conv2d(start*2, start*4, 5, bias=False, padding=2),  # 32*24*24->64*24*24
            nn.BatchNorm2d(start*4, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 64*24*24->64*12*12
            nn.Conv2d(start*4, start*8, 5, bias=False, padding=2),  # 64*12*12->128*12*12
            nn.BatchNorm2d(start*8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 128*12*12->128*6*6
            nn.Conv2d(start*8, start*16, 5, bias=False, padding=2), # 128*6*6->256*6*6
            nn.BatchNorm2d(start*16, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 256*6*6->256*3*3
            nn.Flatten(),                                           # 256*3*3->2304
            nn.Linear(start*16*9, self.args.latent_dim, bias=False)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class pretrain_autoencoder_96_100(nn.Module):
    def __init__(self, args):
        super(pretrain_autoencoder_96_100, self).__init__()
        self.args = args
        start=16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, start, 5, bias=False, padding=2),          # 1*96*96->16*96*96
            nn.BatchNorm2d(start, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 16*96*96->16*48*48
            nn.Conv2d(start, start*2, 5, bias=False, padding=2),    # 16*48*48->32*48*48
            nn.BatchNorm2d(start*2, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 32*48*48->32*24*24
            nn.Conv2d(start*2, start*4, 5, bias=False, padding=2),  # 32*24*24->64*24*24
            nn.BatchNorm2d(start*4, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 64*24*24->64*12*12
            nn.Conv2d(start*4, start*8, 5, bias=False, padding=2),  # 64*12*12->128*12*12
            nn.BatchNorm2d(start*8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 128*12*12->128*6*6
            nn.Conv2d(start*8, start*16, 5, bias=False, padding=2), # 128*6*6->256*6*6
            nn.BatchNorm2d(start*16, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                                      # 256*6*6->256*3*3
            nn.Flatten(),                                           # 256*3*3->2304
            nn.Linear(start*16*9, self.args.latent_dim, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.args.latent_dim, start*16*9, bias=False),
            nn.Unflatten(1, (start*16, 3, 3)),                                  # 2304->256*3*3
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                                        # 256*3*3->256*6*6
            nn.ConvTranspose2d(start*16, start*8, 5, bias=False, padding=2),    # 256*6*6->128*6*6
            nn.BatchNorm2d(start*8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                                        # 128*6*6->128*12*12
            nn.ConvTranspose2d(start*8, start*4, 5, bias=False, padding=2),     # 128*12*12->64*12*12
            nn.BatchNorm2d(start*4, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                                        # 64*12*12->64*24*24
            nn.ConvTranspose2d(start*4, start*2, 5, bias=False, padding=2),     # 64*24*24->32*24*24
            nn.BatchNorm2d(start*2, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                                        # 32*24*24->32*48*48
            nn.ConvTranspose2d(start*2, start, 5, bias=False, padding=2),       # 32*48*48->16*48*48
            nn.BatchNorm2d(start, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                                        # 16*48*48->16*96*96
            nn.ConvTranspose2d(start, 1, 5, bias=False, padding=2),             # 16*96*96->1*96*96
        )        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded