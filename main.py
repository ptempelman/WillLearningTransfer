from dataloader import dataset_vegetable, load_vegetable
from model import resnet18


if __name__ == "__main__":
    # TODO: initialize dataloader
    train_dataset, val_dataset, test_dataset = dataset_vegetable()
    train_dataloader, val_dataloader, test_dataloader = load_vegetable(
        train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=6
    )

    model = resnet18(pretrained=False)

    # TODO: create training loop

    # TODO: train model on easy dataset

    # TODO: adapt model for hard dataset (last layer, maybe first layer)

    # TODO: train again with pretrained model

    # TODO: plot learning curve: no pretraining
    # TODO: plot learning curve: with pretraining
    # TODO: plot learning curves: with pretraining and decreasing amounts of hard dataset data