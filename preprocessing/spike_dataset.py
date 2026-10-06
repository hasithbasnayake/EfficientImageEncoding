import torch
from torch.utils.data import Dataset

class spike_dataset(Dataset):
    """
    A subclass of PyTorch's dataset. Serves latency-encoded images as spike trains

    Parameters
    ----------
    latency_images : torch.Tensor
        An Int tensor of the shape [N, C, H, W] with values representing normalized
        intensities derived from pixel values. Intended to be calculated by latency().
        NOTE: The num_steps argument passed into class instances should be the same as
        the num_steps parameter used for latency().
    images : torch.Tensor
        An image tensor of the shape [N, C, H, W] with values representing pixel intensities.
        NOTE: Should be the original dataset passed into latency()
    labels : torch.Tensor
        A tensor of the shape [N] with ints representing the correct label/category of each image
    num_steps : int
        The number of timesteps in the built spike trains

    """

    def __init__(self, latency_images: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, num_steps: int = 255) -> None:
        self.latency_images = latency_images
        self.images = images
        self.labels = labels
        self.num_steps = num_steps

    def __len__(self):
        return len(self.latency_images.shape[0])

    def __getitem__(self, index):
        latency_image = self.latency_images[index]

        steps = torch.arange(self.num_steps)
        spikes = steps[:, None] == latency_image[None, :]

        data_tuple = (spikes.float(), self.images[index], self.labels[index])
        return data_tuple

    def __str__(self):
        return f"Spike dataset of {len(self)}"

    def plot(self, index):
        return "Unimplemented plot method"