import torch
import easydict
from code.dataloader import make_augmented_dataloader, make_dataloader
from code.train import TrainerDeepSVDD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = easydict.EasyDict({'num_epochs': 100,
                          'num_epochs_ae': 100,
                          'lr': 1e-3,
                          'lr_ae': 1e-2,
                          'weight_decay': 5e-7,
                          'weight_decay_ae': 5e-4,  # default = 5e-3
                          'lr_milestones': [50],  # default = 50
                          'batch_size': 32,
                          'pretrain': True,
                          'net_name': 'LSTM',
                          'num_filter': 128,
                          'latent_dim': 1024,
                          'image_height': 1000,
                          'image_width': 406,
                          'num_workers_dataloader': 6,
                          'output_path': './output/1122_test_08_LSTM/'
                          })
#   'net_name': 'CNN',
#   'output_path': './output/1117_test_05_filter_128_latent_dim_128/',

if __name__ == '__main__':
    print(args.output_path)
    # Train/Test Loader 불러오기
    dataloader_train = make_augmented_dataloader(data_dir='../image/1026_train', args=args, shuffle=True)
    dataloader_test = make_dataloader(data_dir='../image/1026_test', args=args, shuffle=False)

    # Network 학습준비, 구조 불러오기
    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, dataloader_test, device)

    # DeepSVDD를 위한 DeepLearning pretrain 모델로 Weight 학습
    if args.pretrain:
        deep_SVDD.pretrain()

    # 학습된 가중치로 Deep_SVDD모델 Train
    deep_SVDD.train()
