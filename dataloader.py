from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_dataloader(data_dir, args, shuffle):
    data_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Resize((args.image_size,args.image_size))])
    dataset = datasets.ImageFolder(data_dir, transform=data_transform)
    # test_num = len(dataset) - train_num
    # train, test = random_split(dataset, [train_num, test_num])
    # train = datasets.ImageFolder(data_dir, transform=data_transform)
    # test = 
    num_workers = 8
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=shuffle)
    # train_dataloader = DataLoader(train, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    # test_dataloader = DataLoader(test, batch_size=args.batch_size, num_workers=num_workers)
    # return train_dataloader, test_dataloader
    return dataloader

