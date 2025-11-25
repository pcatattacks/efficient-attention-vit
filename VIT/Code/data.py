# data.py

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


def prepare_data_cifar10(batch_size=4, num_workers=2,
                         train_sample_size=None, test_sample_size=None):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (32, 32),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = Subset(trainset, indices)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = Subset(testset, indices)

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    classes = (
        "plane", "car", "bird", "cat",
        "deer", "dog", "frog", "horse", "ship", "truck"
    )

    return trainloader, testloader, classes

def prepare_data_imagenette(
    root="./data/imagenette2-160",
    batch_size=64,
    num_workers=4,
    image_size=160,
    train_sample_size=None,
    val_sample_size=None,
):
    """
    Assumes directory:
        root/
          train/
            class_1/ *.jpg
            ...
          val/
            class_1/ *.jpg
            ...
    """

    train_dir = f"{root}/train"
    val_dir = f"{root}/val"

    # Standard ImageNet-style augmentations but sized for 160x160
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    trainset = ImageFolder(train_dir, transform=train_transform)
    valset = ImageFolder(val_dir, transform=val_transform)

    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = Subset(trainset, indices)

    if val_sample_size is not None:
        indices = torch.randperm(len(valset))[:val_sample_size]
        valset = Subset(valset, indices)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Class names in label order
    classes = trainset.dataset.classes if isinstance(trainset, Subset) else trainset.classes

    return trainloader, valloader, classes
