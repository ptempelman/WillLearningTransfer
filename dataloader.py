import warnings
import torchvision
from torchvision import transforms
import os.path as osp
from torch.utils.data import DataLoader, random_split
from PIL import Image

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)

# TODO: create dataset

# TODO: create dataloader

# TODO: later: apply data augmentation


def create_dataset_vegetable():
    train_dir = osp.join(".data", "vegetable")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)

    return dataset


def load_vegetable(dataset, batch_size, num_workers=2):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def create_dataset_fruit():
    train_dir = osp.join(".data", "fruit")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)

    return dataset


def load_fruit(dataset, batch_size, ignored_ratio, num_workers=2):
    ignored_size = int(ignored_ratio * len(dataset))
    rest_size = len(dataset) - ignored_size

    _, used_dataset = random_split(dataset, [ignored_size, rest_size])

    train_size = int(0.8 * len(used_dataset))
    val_size = len(used_dataset) - train_size
    train_dataset, val_dataset = random_split(used_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
