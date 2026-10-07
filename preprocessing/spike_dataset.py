import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class SpikeDataset(Dataset):
    """
    A subclass of PyTorch's dataset. Serves latency-encoded images as spike trains.

    NOTE: Latencies are stored as spike times by class instances and expanded into binary
    spike trains of the shape [S, F] when indexed. Where S is the length of the spike train,
    provided by num_steps and F is the number of features.

    Parameters
    ----------
    latency_images : torch.Tensor
        An Int tensor of the shape [N, F] with values representing normalized
        intensities derived from pixel values. Each value is the timestep
        at which that input fires, in [0, num_steps - 1]. Zero-intensity
        pixels fire at the last step, num_steps - 1.

        NOTE: The num_steps argument passed during initialization should be the same as
        the num_steps parameter used for latency().
    images : torch.Tensor
        An image tensor of the shape [N, C, H, W] with values representing pixel intensities.

        NOTE: Should be the original dataset passed into latency(), returned as is, used as reconstruction targets.
    labels : torch.Tensor
        Int tensor of shape [N] holding each image's class label.
    num_steps : int
        Length of each spike train. As stated above, must equal the num_steps used in
        latency(), or spikes will be dropped.

    __getitem__
    ----------
    Indexing returns a tuple (spikes, image, label):
        spikes : float tensor of shape [S, F]
        image  : the corresponding entry of `images`
        label  : the corresponding entry of `labels`
    """

    def __init__(self, latency_images: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, num_steps: int = 255) -> None:
        self.latency_images = latency_images
        self.images = images
        self.labels = labels
        self.num_steps = num_steps

    def __len__(self):
        return self.latency_images.shape[0]

    def __getitem__(self, index):
        latency_image = self.latency_images[index]

        steps = torch.arange(self.num_steps)
        spikes = steps[:, None] == latency_image[None, :]

        data_tuple = (spikes.float(), self.images[index], self.labels[index])
        return data_tuple

    def __str__(self):
        return f"Spike dataset containing ({len(self)}) spike-encoded images in the shape [{self.num_steps}, {self.latency_images.size(dim=1)}]"

    def plot(self, index=None):
        if index is None:
            index = 0

        spikes, image, label = self[index]
        counts = spikes.sum(dim=1)
        x, y = spikes.size()

        print(f"Shape of spikes: {spikes.shape}")
        print(f"Shape of image: {image.size()}")
        print(f"Label: {label}")

        # NOTE: Make sure that the image is the form [C, H, W]
        num_channels = image.size(dim=0)

        fig, axs = plt.subplots(1, (num_channels + 1), figsize=(4 * (num_channels + 1), 4), layout='constrained')
        axs[0].plot(counts)
        axs[0].set_title('Spikes')
        axs[0].set_xlabel('Timesteps')
        axs[0].set_ylabel('Count')
        axs[0].set_xlim(0, x)
        axs[0].set_ylim(0, y)

        for channel in range(num_channels):
            channel_image = image[channel]
            ax = axs[channel + 1]
            ax.imshow(channel_image, cmap="gray")
            ax.set_title(f'Image label: {label}')
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Pixels')

        plt.show()



