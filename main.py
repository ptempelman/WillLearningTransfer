from dataloader import create_dataset_vegetable, load_vegetable
from model import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(model, train_loader, val_loader):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = model.to(device)
    num_epochs = 10
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


if __name__ == "__main__":
    print("Creating dataset & dataloader")
    dataset_vegetable = create_dataset_vegetable()
    train_loader, val_loader = load_vegetable(
        dataset_vegetable, batch_size=64, num_workers=4
    )
    
    print("Initializing model")
    model = resnet18(
        pretrained=False,
        num_classes=len(dataset_vegetable.classes),
    )

    print("Start training")
    train(model, train_loader, val_loader)

    # TODO: adapt model for hard dataset (last layer, maybe first layer)

    # TODO: train again with pretrained model

    # TODO: plot learning curve: no pretraining
    # TODO: plot learning curve: with pretraining
    # TODO: plot learning curves: with pretraining and decreasing amounts of hard dataset data
