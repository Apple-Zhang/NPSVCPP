import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision import models
from torch import nn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def use_data(data_name: str, batch_size=64):
    data_name = data_name.lower()
    if data_name == "cifar10":
        transform = transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        cifar10 = datasets.CIFAR10('./data/', train=True, download=True, transform=transform)
        train_loader = data.DataLoader(cifar10, batch_size=batch_size, shuffle=True) 
        cifar10 = datasets.CIFAR10('./data/', train=False, download=True, transform=transform)
        test_loader  = data.DataLoader(cifar10, batch_size=batch_size, shuffle=False)
        num_classes = 10
    elif data_name == "flowers102":
        transform = transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        flower = datasets.Flowers102("./data/", split="train", download=True, transform=transform)
        train_loader = data.DataLoader(flower, batch_size=batch_size, shuffle=True)
        flower = datasets.Flowers102("./data/", split="test", download=True, transform=transform)
        test_loader  = data.DataLoader(flower, batch_size=batch_size, shuffle=False)
        num_classes = 102
    elif data_name == "dtd":
        transform = transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        place = datasets.DTD("./data/", split="train", download=True, transform=transform)
        train_loader = data.DataLoader(place, batch_size=batch_size, shuffle=True)
        place = datasets.DTD("./data/", split="test", download=True, transform=transform)
        test_loader  = data.DataLoader(place, batch_size=batch_size, shuffle=False)
        num_classes = 47
    else:
        raise ValueError("Unknown dataset: %s" % data_name)
    
    return train_loader, test_loader, num_classes

def use_fea_extractor(model_name: str):
    model_name = model_name.lower()

    dim_out = None

    if model_name == "resnet18":
        fea_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        fea_extractor.fc = nn.Identity()
        dim_out = 512
    elif model_name == "resnet34":
        fea_extractor = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        fea_extractor.fc = nn.Identity()
        dim_out = 512
    elif model_name == "resnet50":
        fea_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        fea_extractor.fc = nn.Identity()
        dim_out = 2048
    elif model_name == "resnet101":
        fea_extractor = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        fea_extractor.fc = nn.Identity()
        dim_out = 2048

    return fea_extractor, dim_out