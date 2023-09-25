import torch
import easydict
from dataloader import make_dataloader
from train import TrainerDeepSVDD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = easydict.EasyDict({
        'num_epochs':100,
        'num_epochs_ae':100,
        'lr':1e-3,
        'lr_ae':1e-3,
        'weight_decay':5e-7,
        'weight_decay_ae':5e-3,
        'lr_milestones':[50],
        'batch_size':1024,
        'pretrain':True,
        'path':'./0830_96_100_clip/',
        'net_name':1024,
        'image_size':1024,
        'num_filter':32,
        'latent_dim':100
                })

if __name__ == '__main__':
    print(args.path)
    # Train/Test Loader 불러오기
    dataloader_train = make_dataloader(data_dir='../image/0830', args=args, shuffle=True)
    # dataloader_train = make_dataloader(data_dir='../image/janggi/noise', args=args, shuffle=True)
    # dataloader_test = make_dataloader(data_dir='../image/janggi/event', args=args, shuffle=False)

    # Network 학습준비, 구조 불러오기
    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)

    # DeepSVDD를 위한 DeepLearning pretrain 모델로 Weight 학습
    if args.pretrain:
        deep_SVDD.pretrain()

    # 학습된 가중치로 Deep_SVDD모델 Train
    net, c = deep_SVDD.train()
    # torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()}, args.path + 'finish_svdd.pth')
