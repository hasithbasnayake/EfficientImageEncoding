import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from preprocessing.latency import latency
from preprocessing.spike_dataset import SpikeDataset
from preprocessing.difference_of_guassians import gen_lgn_receptive_fields, difference_of_gaussians

import numpy as np
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    shape = torch.tensor([6,6])
    ctr = (1/3) * shape[0]
    sur = (2/3) * shape[0]

    training_set = torchvision.datasets.FashionMNIST(root="assets/data/", train=True, transform=transforms.ToTensor(), download=True)

    training_images = training_set.data.unsqueeze(1)
    training_labels = training_set.targets
    training_images = difference_of_gaussians(training_images, shape, ctr, sur, threshold=1/255)

    latency_images = latency(training_images)
    latency_images = latency_images.flatten(start_dim=1)

    spike_set = SpikeDataset(latency_images=latency_images, images=training_images, labels=training_labels, num_steps=255)
    loader = DataLoader(spike_set, batch_size=1, shuffle=True)

    spike_set.plot(0)