from model import resnet18


if __name__ == "__main__":
    # TODO: initialize dataloader

    model = resnet18(pretrained=False)

    # TODO: create training loop

    # TODO: train model on easy dataset

    # TODO: adapt model for hard dataset (last layer, maybe first layer)

    # TODO: train again with pretrained model

    # TODO: plot learning curve: no pretraining
    # TODO: plot learning curve: with pretraining
    # TODO: plot learning curves: with pretraining and decreasing amounts of hard dataset data
