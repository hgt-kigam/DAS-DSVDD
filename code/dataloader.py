from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_dataloader(data_dir, args, shuffle):
    data_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers_dataloader, shuffle=shuffle)
    return dataloader

