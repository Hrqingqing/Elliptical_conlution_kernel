import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


#
def get10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, augmentation="yes", val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        if augmentation == "yes":
            print('Cifar10: Using data augmentation.')
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root=data_root, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=True, **kwargs)
            ds.append(train_loader)
            print(f"loading using: {len(ds)}")
        if augmentation == "no":
            print('Cifar10: Not using data augmentation.')
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root=data_root, train=True, download=True,
                    transform=transforms.Compose([
                        # transforms.Pad(4),
                        # transforms.RandomCrop(32),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=True, **kwargs)
            ds.append(train_loader)
            print(len(ds))
            print(f"loading not using: {len(ds)}")
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
        print(f"loading val: {len(ds)}")
    ds = ds[0] if len(ds) == 1 else ds
    print(f"loading all: {len(ds)}")
    return ds


def getImageNet(batch_size, data_root='/data2/imagenet-data',
                **kwargs):
    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    num_workers = kwargs.setdefault('num_workers', 1)
    pin_memory = True
    kwargs.pop('input_size', None)
    print("Building ImageNet data loader with {} workers".format(num_workers))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader
