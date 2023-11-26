import torchvision
from torchvision import transforms
import os.path as osp
from torch.utils.data import DataLoader

# TODO: create dataset

# TODO: create dataloader

# TODO: later: apply data augmentation


def dataset_vegetable():
    train_dir = osp.join(".data", "vegetable", "train")
    val_dir = osp.join(".data", "vegetable", "validation")
    test_dir = osp.join(".data", "vegetable", "test")

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.1, hue=0.1, saturation=0.1
            ),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(1, 2), shear=15
            ),
            transforms.GaussianBlur(kernel_size=(5, 9)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir, transform=train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_dir, transform=test_transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_dir, transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset


def load_vegetable(train_dataset, val_dataset, test_dataset, batch_size, num_workers=2):
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataloader, val_dataloader, test_dataloader
