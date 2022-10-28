import torch
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [5, 5]

num_clients = 5

train_dataset = MNIST('./data/working', train=True, download=True, transform=transforms.ToTensor())
test_dataset = MNIST('./data/working', train=False, download=True, transform=transforms.ToTensor())

train_dataset, dev_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.83), int(len(train_dataset) * 0.17)])

# Define Notebook Constants
total_train_size = len(train_dataset)
total_test_size = len(test_dataset)
total_dev_size = len(dev_dataset)

examples_per_client = total_train_size // num_clients
client_datasets = random_split(train_dataset, [min(i + examples_per_client, 
           total_train_size) - i for i in range(0, total_train_size, examples_per_client)])


for i in range(len(client_datasets)):
    file_name = "client_dataset_0" + str(i+1)
    torch.save(client_datasets[0], './data_tensor/' + file_name)