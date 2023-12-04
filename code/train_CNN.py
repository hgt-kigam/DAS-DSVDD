import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from .network.main import DeepSVDD, Autoencoder
from torch.utils.tensorboard import SummaryWriter


class TrainerDeepSVDD:
    def __init__(self, args, data_loader, dataloader_test, device):
        self.args = args
        self.train_loader = data_loader
        self.dataloader_test = dataloader_test
        self.device = device
        os.mkdir(self.args.output_path)
        self.writer = SummaryWriter(log_dir=self.args.output_path)

    def plot_classes_preds(self, net):
        net.eval()
        with torch.no_grad():
            for x_cpu, _ in self.dataloader_test:
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(4, 10))
                x = x_cpu.float().to(self.device)
                x_hat = net(x).detach().cpu()
                axs[0][0].imshow(x_cpu[0, 0, :, :], cmap='gray')
                axs[0][0].set_title('input')
                axs[1][0].imshow(x_cpu[1, 0, :, :], cmap='gray')
                axs[1][0].set_title('input')
                axs[0][1].imshow(x_hat[0, 0, :, :], cmap='gray')
                axs[0][1].set_title('output')
                axs[1][1].imshow(x_hat[1, 0, :, :], cmap='gray')
                axs[1][1].set_title('output')
                net.train()
                break
        return fig

    def pretrain(self):
        ae = Autoencoder(self.args).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(),
                                     lr=self.args.lr_ae,
                                     weight_decay=self.args.weight_decay_ae
                                     )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones,
                                                         gamma=0.1
                                                         )
        self.writer.add_graph(ae, torch.Tensor(1, 1, self.args.image_height, self.args.image_width).to(self.device))
        print(summary(ae, (1, self.args.image_height, self.args.image_width)))
        ae.train()

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
            self.writer.add_scalar('Loss/AutoEncoder', loss_temp, epoch+1)
            losses.append(loss_temp)
            self.writer.add_figure('actuals vs. predictions', self.plot_classes_preds(ae), global_step=epoch+1)
            if loss_temp <= min(losses):
                c = self.set_c(ae, self.train_loader)
                torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': ae.state_dict()},
                           self.args.output_path+'AE_best_save.pth'
                           )
                ae.train()
                print(f'save best model at {epoch+1} epoch')
            print(f'Pretraining Autoencoder... Epoch: {epoch+1}, Loss: {loss_temp:.6f}, Time: {time.time()-start}')
        self.writer.flush()
        self.writer.close()
        losses = np.array(losses)
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.plot(losses)
        plt.savefig(self.args.output_path+'AE_Loss.png', format='png')
        np.save(self.args.output_path+'AE_Loss.npy', losses)
        self.save_weights_for_DeepSVDD(ae)

    def save_weights_for_DeepSVDD(self, model):
        state_dict = torch.load(self.args.output_path+'AE_best_save.pth')
        model.load_state_dict(state_dict['net_dict'])
        c = torch.Tensor(state_dict['center']).to(self.device)
        net = DeepSVDD(self.args).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()},
                   self.args.output_path+'pretrained_SVDD.pth'
                   )

    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        # print('z_ -after torch.cat', z_.shape)
        c = torch.mean(z_, dim=0)
        # print('c -after torch.mean', c.shape)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self):
        net = DeepSVDD(self.args).to(self.device)
        if self.args.pretrain is True:
            state_dict = torch.load(self.args.output_path+'pretrained_SVDD.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        print(summary(net, (1, self.args.image_height, self.args.image_width)))
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
                net.eval()
                torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()},
                           self.args.output_path+'Deep_SVDD_best_save.pth'
                           )
                net.train()
                print(f'save best model at {epoch+1} epoch')
            print(f'Training Deep SVDD... Epoch: {epoch+1}, Loss: {loss_temp:.6f}, Time: {time.time()-start}')
        losses = np.array(losses)
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.plot(losses)
        plt.savefig(self.args.output_path+'Deep_SVDD_Loss.png', format='png')
        np.save(self.args.output_path+'Deep_SVDD_Loss.npy', losses)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)