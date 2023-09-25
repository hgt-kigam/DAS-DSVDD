import numpy as np
import os
import time
import torch
from torchsummary import summary
from network.main import Deep_SVDD, Autoencoder

class TrainerDeepSVDD:
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device
        os.mkdir(self.args.path)

    def pretrain(self):
        ae = Autoencoder(self.args).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        ae.train()
        print(summary(ae, (1, self.args.net_name, self.args.net_name)))
        losses = []
        for epoch in range(self.args.num_epochs_ae):
            start = time.time()
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            loss_temp = total_loss/len(self.train_loader)
            losses.append(loss_temp)
            if loss_temp <= min(losses):
                self.ae_best_model_save(ae, self.train_loader)
                print(f'save best model at {epoch+1} epoch')
            print(f'Pretraining Autoencoder... Epoch: {epoch+1}, Loss: {loss_temp:.6f}, Time: {time.time()-start}')
        losses = np.array(losses)
        np.save(self.args.path+'AE_Loss.npy', losses)
        self.save_weights_for_DeepSVDD(ae)
    
    def ae_best_model_save(self, model, dataloader):
        c = self.set_c(model, dataloader)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': model.state_dict()},
                    self.args.path+'AE_best_save.pth')
        
    def save_weights_for_DeepSVDD(self, model):
        """
        학습된 AutoEncoder의 가중치를 Deep_SVDD 모델에 Initialize해주는 함수
        Best_model로 적용하려고 코드가 좀 복잡해짐
        train함수의 if self.args.pretrain==True 하위 명령어를 참고했음
        왜 구분해놓은지는 모르겠음 load_state_dict의 strict인수때문인가?
        이해도가 높아져서 내 예상이 맞다면 이 함수를 없애고 train함수 내부로 아래 명령어를 편입하는게 좋을듯
        """
        state_dict = torch.load(self.args.path+'AE_best_save.pth')
        model.load_state_dict(state_dict['net_dict'])
        c = torch.Tensor(state_dict['center']).to(self.device)
        net = Deep_SVDD(self.args).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()},
                    self.args.path+'pretrained_SVDD.pth')

    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self):
        net = Deep_SVDD(self.args).to(self.device)
        if self.args.pretrain==True:
            state_dict = torch.load(self.args.path+'pretrained_SVDD.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        print(summary(net, (1, self.args.net_name, self.args.net_name)))
        net.train()
        losses = []
        for epoch in range(self.args.num_epochs):
            start = time.time()
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            loss_temp = total_loss/len(self.train_loader)
            losses.append(loss_temp)
            if loss_temp <= min(losses):
                torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()},
                    self.args.path+'Deep_SVDD_best_save.pth')
                print(f'save best model at {epoch+1} epoch')
            print(f'Training Deep SVDD... Epoch: {epoch+1}, Loss: {loss_temp:.6f}, Time: {time.time()-start}')
        losses = np.array(losses)
        np.save(self.args.path+'Deep_SVDD_Loss.npy', losses)
        self.net = net
        self.c = c

        return self.net, self.c

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)