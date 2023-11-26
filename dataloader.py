import torchvision
from torchvision import transforms
import os.path as osp
from torch.utils.data import DataLoader, random_split

# TODO: create dataset

# TODO: create dataloader

# TODO: later: apply data augmentation


def create_dataset_vegetable():
    train_dir = osp.join(".data", "vegetable")

    # train_transform = transforms.Compose(
    #     [
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomRotation(15),
    #         transforms.ColorJitter(
    #             brightness=0.2, contrast=0.1, hue=0.1, saturation=0.1
    #         ),
    #         transforms.RandomAffine(
    #             degrees=15, translate=(0.1, 0.1), scale=(1, 2), shear=15
    #         ),
    #         transforms.GaussianBlur(kernel_size=(5, 9)),
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #     ]
    # )

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
