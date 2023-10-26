import torch
import easydict
from code.dataloader import make_augmented_dataloader
from code.train import TrainerDeepSVDD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = easydict.EasyDict({
    'num_epochs': 500,
    'num_epochs_ae': 500,
    'lr': 1e-3,
    'lr_ae': 1e-3,
    'weight_decay': 5e-7,
    'weight_decay_ae': 5e-3,
    'lr_milestones': [50],
    'batch_size': 16,
    'pretrain': True,
    'net_name': 406,
    'num_filter': 32,
    'latent_dim': 100,
    'image_height': 1000,
    'image_width': 406,
    'num_workers_dataloader': 4,
    'output_path': './output/1026_Newdata_Augment_test/'
})

if __name__ == '__main__':
    print(args.output_path)
    # Train/Test Loader 불러오기
    dataloader_train = make_augmented_dataloader(data_dir='../image/1026_train', args=args, shuffle=True)

    # Network 학습준비, 구조 불러오기
    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)

    # DeepSVDD를 위한 DeepLearning pretrain 모델로 Weight 학습
    if args.pretrain:
        deep_SVDD.pretrain()

    # 학습된 가중치로 Deep_SVDD모델 Train
    deep_SVDD.train()
