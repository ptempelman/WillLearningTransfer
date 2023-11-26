import itertools
from dataloader import (
    create_dataset_fruit,
    create_dataset_vegetable,
    load_fruit,
    load_vegetable,
)
from model import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os.path as osp


def train(model, train_loader, val_loader, pretrain=False, num_epochs=10):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        if pretrain:
            save_checkpoint(
                model,
                epoch,
                osp.join(
                    "./.checkpoints", f"checkpoint_epoch_{epoch+1}_loss_{val_loss}.pth"
                ),
            )


def save_checkpoint(model, epoch, checkpoint_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def pretrain(epochs):
    print("Creating vegetable dataset & dataloader")
    dataset_vegetable = create_dataset_vegetable()
    train_loader, val_loader = load_vegetable(
        dataset_vegetable, batch_size=64, num_workers=4
    )

    print("Initializing model")
    model = resnet18(
        pretrained=False,
        num_classes=len(dataset_vegetable.classes),
    )

    print("Start pretraining")
    train(model, train_loader, val_loader, pretrain=True, num_epochs=epochs)


if __name__ == "__main__":
    # pretrain(epochs=10)

    combinations = list(itertools.product([False, True], [0, 0.2, 0.4, 0.6, 0.8]))

    for pretrain, ignore_ratio in combinations:
        print(pretrain, ignore_ratio)

        print("Creating fruit dataset & dataloader")
        dataset_fruit = create_dataset_fruit()
        train_loader, val_loader = load_fruit(
            dataset_fruit, batch_size=16, ignored_ratio=ignore_ratio, num_workers=4
        )

        print("Initializing model")
        model = resnet18(
            pretrained=False,
            num_classes=len(dataset_fruit.classes),
        )
        if pretrain:
            print("Loading in pretrained model")
            checkpoint_path = (
                ".checkpoints/checkpoint_epoch_10_loss_0.05014692123692769.pth"
            )
            model = load_checkpoint(model, checkpoint_path)
            model.fc = nn.Linear(
                in_features=model.fc.in_features,
                out_features=len(dataset_fruit.classes),
            )

        print("Start training")
        train(model, train_loader, val_loader, pretrain=False, num_epochs=10)

    # TODO: plot learning curve: no pretraining
    # TODO: plot learning curve: with pretraining
    # TODO: plot learning curves: with pretraining and decreasing amounts of hard dataset data
