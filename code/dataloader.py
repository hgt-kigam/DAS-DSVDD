from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_dataloader(data_dir, args, shuffle):
    data_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, data_transform)
    # dataloader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.num_workers_dataloader)
    dataloader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.num_workers_dataloader)
    return dataloader


def make_autoaugmented_dataloader(data_dir, args, shuffle):
    data_transform = transforms.Compose([transforms.Grayscale(), transforms.AutoAugment(), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, data_transform)
    dataloader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.num_workers_dataloader)
    return dataloader


def make_augmented_dataloader(data_dir, args, shuffle):
    data_transform = transforms.Compose([transforms.Grayscale(),
                                         transforms.RandomApply([transforms.RandomVerticalFlip(0.2),
                                                                 transforms.RandomHorizontalFlip(0.2),
                                                                 transforms.RandomInvert(0.5),
                                                                 transforms.RandomRotation(180),
                                                                 transforms.RandomAutocontrast()]),
                                         transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, data_transform)
    dataloader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.num_workers_dataloader, drop_last=True)
    return dataloader
