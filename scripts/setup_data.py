import torchvision as tv

tv.datasets.CIFAR10("data",download=True,train=True)
tv.datasets.CIFAR10("data",download=True,train=False)
