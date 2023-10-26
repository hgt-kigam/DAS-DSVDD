import torch
import easydict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from code.dataloader import make_dataloader
from code.network.DSVDD_rectangle import Deep_SVDD


def test_eval(net, c, device, dataloader):

    scores = []
    labels = []
    x_list = []
    z_list = []
    net.eval()
    print('Testing...')
    with torch.no_grad():

        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.mean((z - c) ** 2, dim=1)

            x_list.append(x.detach().cpu())
            z_list.append(((z-c)**2).detach().cpu())
            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    x_list = torch.cat(x_list).numpy()
    z_list = torch.cat(z_list).numpy()
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()

    return labels, scores, z_list, x_list


def plot_representations(data, labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
    handles, labels = scatter.legend_elements()
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)
    # legend = ax.legend(handles=handles, labels=labels)
    ax.legend(handles=handles, labels=labels)
    plt.show()


def get_tsne(data, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=908)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = easydict.EasyDict({
       'batch_size': 1024,
       'latent_dim': 100,
       'path': './1020_AutoAugment_test/',
       'num_workers_dataloader': 8,
       'num_filter': 32
       })

dataloader_test = make_dataloader(data_dir='../image/1010_test', args=args, shuffle=False)
# dataloader_test = make_dataloader(data_dir='../image/0830/', args=args, shuffle=False)
# dataloader_test = make_dataloader(data_dir='../image/janggi/noise', args=args, shuffle=False)

# state_dict = torch.load(args.path + 'pretrained_SVDD.pth')
state_dict = torch.load(args.path + 'Deep_SVDD_best_save.pth')
net = Deep_SVDD(args).to(device)
net.load_state_dict(state_dict['net_dict'])
c = torch.Tensor(state_dict['center']).to(device)

if __name__ == '__main__':
    # scores = []
    # labels = []
    # net.eval()
    # print('Testing...')
    # with torch.no_grad():
    #     for x, y in dataloader_test:
    #         x = x.float().to(device)
    #         z = net(x)
    #         score = torch.sum((z - c) ** 2, dim=1)

    #         scores.append(score.detach().cpu())
    #         labels.append(y.cpu())
    # labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()

    labels, scores, z_list, x_list = test_eval(net, c, device, dataloader_test)
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    # x_list = x_list.reshape(2033,224,224)
    print(x_list.shape, z_list.shape)
    # for i in range(labels.shape[0]):
    #     print(labels[i], scores[i])
    # tsne_data = get_tsne(x_list)
    # plot_representations(tsne_data, labels)
    tsne_data = get_tsne(z_list)
    plot_representations(tsne_data, labels)
