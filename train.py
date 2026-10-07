import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.dataset import _T_co

from preprocessing.latency import latency
from preprocessing.spike_dataset import SpikeDataset

if __name__ == '__main__':
    training_set = torchvision.datasets.FashionMNIST(root="assets/data/", train=True, transform=transforms.ToTensor(), download=True)

    training_images = training_set.data.unsqueeze(1)
    training_labels = training_set.targets

    latency_images = latency(training_images)
    latency_images = latency_images.flatten(start_dim=1)

    spike_set = SpikeDataset(latency_images=latency_images, images=training_images, labels=training_labels, num_steps=255)
    print(spike_set)
    print(spike_set[0][0].size())