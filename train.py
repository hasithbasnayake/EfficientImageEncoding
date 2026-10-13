import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from preprocessing.latency import latency
from preprocessing.spike_dataset import SpikeDataset
from preprocessing.difference_of_guassians import lgn_preprocessing

from model.model import Net
from model.learning import update_weights, wta

import numpy as np
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    shape = torch.tensor([6,6])
    ctr = (1/3) * shape[0]
    sur = (2/3) * shape[0]

    A_plus = 5e-3
    A_minus = 3.75e-3
    tau = 200
    mu_plus = 0.65
    mu_minus = 0.05
    params = (A_plus, A_minus, tau, mu_plus, mu_minus)

    model = Net(
        num_input=1568,
        num_output=100,
        beta=.9, #.9
        threshold=82, #20
        reset_mechanism="zero",
        seed=0
    )

    training_set = torchvision.datasets.FashionMNIST(root="assets/data/", train=True, transform=transforms.ToTensor(), download=True)

    training_images = training_set.data.unsqueeze(1)
    training_labels = training_set.targets
    training_images = lgn_preprocessing(training_images, shape, ctr, sur, threshold=1/255)

    latency_images = latency(training_images)
    latency_images = latency_images.flatten(start_dim=1)

    spike_set = SpikeDataset(latency_images=latency_images, images=training_images, labels=training_labels, num_steps=255)
    loader = DataLoader(spike_set, batch_size=1, shuffle=True)

    W0 = model.fc1.weight.detach().clone()

    for i in range(60000):
        print(f"Image: {i}")
        spikes, images, label = spike_set[i]

        with torch.no_grad():
            spk_rec, mem_rec = model.forward(spikes.squeeze())

        weight_update = update_weights(model.fc1.weight, spikes, spk_rec, 1, True, params)

        with torch.no_grad():
            model.fc1.weight += weight_update
            model.fc1.weight.clamp_(0, 1)

        # fig, ax = plt.subplots(figsize=(10,5))
        # splt.raster(spk_rec, ax, s=5, color="black", marker="|")
        # ax.set_xlim(0, spk_rec.shape[0])
        # ax.set_ylim(-0.5, spk_rec.shape[1] - 0.5)
        # ax.set_xlabel("Timestep")
        # ax.set_ylabel("Neuron")
        # ax.set_title("Output spike raster")
        # plt.show()

        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(mem_rec, lw=0.5, alpha=0.3, color="gray")
        # ax.set_xlim(0, 20)  # zoom to where the winner is decided; remove to see all 255 steps
        # ax.set_xlabel("Timestep")
        # ax.set_ylabel("Membrane potential")
        # ax.set_title("Membrane potential traces")
        # plt.show()
        # plt.show()

    W = model.fc1.weight.detach().cpu()
    W = W - W0
    W = W.reshape(100, 2, 28, 28)

    rf = W[:, 0] - W[:, 1]

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(rf[i], cmap="bwr")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
