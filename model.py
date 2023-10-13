"""
import torch.nn as nn

class Deep_SVDD_224(nn.Module):
    def __init__(self):
        super(Deep_SVDD_224, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 5, bias=False, padding=2),     # 1*224*224->32*224*224
            nn.BatchNorm2d(8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 32*224*224->32*112*112
            nn.Conv2d(8, 16, 5, bias=False, padding=2),    # 32*112*112->64*112*112
            nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 64*112*112->64*56*56
            nn.Conv2d(16, 32, 5, bias=False, padding=2),   # 64*56*56->128*56*56
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 128*56*56->128*28*28
            nn.Conv2d(32, 64, 5, bias=False, padding=2),  # 128*28*28->256*28*28
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 256*28*28->256*14*14
            nn.Conv2d(64, 128, 5, bias=False, padding=2),  # 256*14*14->512*14*14
            nn.BatchNorm2d(128, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 512*14*14->512*7*7
            nn.Flatten(),                                   # 512*7*7->25088
            nn.Linear(6272, 1568, bias=False),
            nn.Linear(1568, 392, bias=False),
            nn.Linear(392, 98, bias=False) #여기 바뀔경우 train.py의 latent_dim 바꿔야함
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class pretrain_autoencoder_224(nn.Module):
    def __init__(self):
        super(pretrain_autoencoder_224, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 5, bias=False, padding=2),      # 1*224*224->8*224*224
            nn.BatchNorm2d(8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 8*224*224->8*112*112
            nn.Conv2d(8, 16, 5, bias=False, padding=2),     # 8*112*112->16*112*112
            nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 16*112*112->16*56*56
            nn.Conv2d(16, 32, 5, bias=False, padding=2),    # 16*56*56->32*56*56
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 32*56*56->32*28*28
            nn.Conv2d(32, 64, 5, bias=False, padding=2),    # 32*28*28->64*28*28
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 64*28*28->64*14*14
            nn.Conv2d(64, 128, 5, bias=False, padding=2),   # 64*14*14->128*14*14
            nn.BatchNorm2d(128, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2,2),                              # 128*14*14->128*7*7
            nn.Flatten(),                                   # 128*7*7->6272
            nn.Linear(6272, 1568, bias=False),
            nn.Linear(1568, 392, bias=False),
            nn.Linear(392, 98, bias=False) #여기 바뀔경우 train.py의 latent_dim 바꿔야함
        )

        self.decoder = nn.Sequential(
            nn.Linear(98, 392, bias=False),
            nn.Linear(392, 1568, bias=False),
            nn.Linear(1568, 6272, bias=False),
            nn.Unflatten(1, (128, 7, 7)),                           # 6272->128*7*7
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                            # 128*7*7->128*14*14
            nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2), # 128*14*14->64*14*14
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                            # 64*14*14->64*28*28
            nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2), # 64*28*28->32*28*28
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                            # 32*28*28->32*56*56
            nn.ConvTranspose2d(32, 16, 5, bias=False, padding=2),  # 32*56*56->16*56*56
            nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                            # 16*56*56->16*112*112
            nn.ConvTranspose2d(16, 8, 5, bias=False, padding=2),   # 16*112*112->8*112*112
            nn.BatchNorm2d(8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                            # 8*112*112->8*224*224
            nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2),    # 8*224*224->1*224*224
        )        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


#############################################################################################################


class Deep_SVDD_96(nn.Module):
    def __init__(self):
        super(Deep_SVDD_96, self).__init__()
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
            nn.Linear(start*16*9, start*4*9, bias=False),
            nn.Linear(start*4*9, start*9, bias=False),
            nn.Linear(start*9, 36, bias=False) #여기 바뀔경우 train.py의 latent_dim 바꿔야함
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class pretrain_autoencoder_96(nn.Module):
    def __init__(self):
        super(pretrain_autoencoder_96, self).__init__()
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
            nn.Linear(start*16*9, start*4*9, bias=False),
            nn.Linear(start*4*9, start*9, bias=False),
            nn.Linear(start*9, 36, bias=False) #여기 바뀔경우 train.py의 latent_dim 바꿔야함
        )

        self.decoder = nn.Sequential(
            nn.Linear(36, start*9, bias=False),
            nn.Linear(start*9, start*4*9, bias=False),
            nn.Linear(start*4*9, start*16*9, bias=False),
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


#############################################################################################################


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
"""
