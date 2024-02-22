import os
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for transformation
import torch  # PyTorch package
import torch.nn as nn  # basic building block for neural neteorks
import torch.nn.functional as F  # import convolution functions like Relu
import torch.optim as optim  # optimzer
import torchvision  # load datasets
import torchvision.transforms as transforms  # transform data
import sys

from test_model import test
from train_model import train

if __name__ == "__main__":
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding = 4, padding_mode = 'reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 200
    num_workers = 5

    train_data = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_data = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    classes = (
        "airplanes",
        "cars",
        "birds",
        "cats",
        "deer",
        "dogs",
        "frogs",
        "horses",
        "ships",
        "trucks",
    )
    match sys.argv[1]:
        case '-train':
            # Train
            train(classes, train_loader) 

            # Testing
            test(test_loader)
            print("Model has been trained")
        case _:
            if os.path.isfile("cifar_net.pth"):
                print("Model has been trained already")
            else:
                # Train
                train(classes, train_loader) 

                # Testing
                test(test_loader)
                print("Model has been trained")
