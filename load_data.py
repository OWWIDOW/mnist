from torchvision import *

def load_trainset():
    train_data = datasets.MNIST(
        './mnist',train=True,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    )
    return train_data

def load_testset():
    test_data = datasets.MNIST(
        './mnist',train=False,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
    )
    return test_data
