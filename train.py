import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
'''
What's the structure of Project Dynapse going to be?

- assets
	- data
		- where the image dataset gets downloaded to
	- example_models
		- contains a pre-trained model that users can test on
	- saved_models
- model folder
	- model.py: defines the model class and the forward pass function, as well as class methods to print out the dataset in various forms (matplotlib, just the words, etc)
- learning folder
	- stdp.py: calculates a weight update given a model, set of output neurons, has a flag to perform WTA inhibition
	- wta.py: a helper function that chooses a neuron and inhibits the rest
- train folder
	- input.py: a thin wrapper that allows a user to kick off a run
	- train.py: contains the training code, should be a thin wrapper that
		- starts by asking for user input
		- a user provides input with a bunch of flags
		- training begins
	- test.py
	- recon.py

what are the actions a user should be able to perform on a model?

A user should be prompted to either load a model from the saved_models by providing a directory and then from there, asked if they want to train, test, and reconstruct from this model.
'''

# A thin wrapper that trains the model on a library of images

# Data Preprocessing
    # Load the image dataset in
    # Split the image dataset into train and test
    # Take all the train images and apply DoG filtering
    # Take all the DoG train images and convert them into spikes
#
# class SpikingDataset(Dataset):
#     '''
#     Instantiates a spiking dataset for use by the PyTorch dataloader class.
#     Takes in spikes of the format [N, T], where N is the number of images,
#     and T is the number of spike trains for each image, the original dataset from
#     training_set.data, as well as the labels.
#
#     For DoG convolved images, this is of the format [N, 1568]
#     For non-convolved images, this is of the format [N, 784]
#
#     The class outputs, using getitem, [255,1568] or whatever timestep is provided
#     It's a tuple ([255,1568], image, label)
#     '''
#
#     def __init__(self, spikes, timesteps=255):
#         self.spikes = spikes
#         self.timesteps = timesteps
#
#     def __len__(self):
#         return self.shape[0]
#
#     def __getitem__(self, idx):
#
#     def __getitem__(self, idx):
#         image = self.spikes
#
#         image = self.raw_img[idx]
#         mask = self.data_mask[idx]
#
#         sample = {'image': image, 'mask': mask}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

Z = [[0.0, 0.0, 0.0, 0.0]] * 4

training_test_tensor = torch.tensor([                         # [5, 1, 4, 4]
    [Z],                                          # 0: blank -> all "no spike"
    [[[0.5, 0.5, 0.5, 0.5],                       # 1: uniform -> min == max (divide-by-zero check)
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5]]],
    [[[0.1, 0.2, 0.3, 0.4],                       # 2: increasing brightness -> decreasing latency
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.0, 0.0, 0.0],                       # 3: single pixel -> one spike at step 0
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0]]],
    [[[0.0, 0.5, 0.5, 0.0],                       # 4: typical image
      [0.5, 1.0, 1.0, 0.5],
      [0.5, 1.0, 1.0, 0.5],
      [0.0, 0.5, 0.5, 0.0]]],
])
trainingDoG_test_tensor = torch.tensor([                         # [5, 2, 4, 4]
    [Z, Z],                                       # 0: both channels empty
    [[[0.2, 0.4, 0.0, 0.0],                       # 1: ON only, OFF empty
      [0.6, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0]], Z],
    [Z, [[0.2, 0.4, 0.0, 0.0],                    # 2: OFF only, ON empty
         [0.6, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]]],
    [[[1.0, 1.0, 0.0, 0.0],                       # 3: ON max 1.0, OFF max 0.5
      [1.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0]],
     [[0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.5, 0.5],
      [0.0, 0.0, 0.5, 0.5]]],
    [[[0.0, 0.5, 0.5, 0.0],                       # 4: identical channels
      [0.5, 1.0, 1.0, 0.5],
      [0.5, 1.0, 1.0, 0.5],
      [0.0, 0.5, 0.5, 0.0]],
     [[0.0, 0.5, 0.5, 0.0],
      [0.5, 1.0, 1.0, 0.5],
      [0.5, 1.0, 1.0, 0.5],
      [0.0, 0.5, 0.5, 0.0]]],
])


def latency(training_images, num_steps=255):
    '''
    Placeholder for docstring.
    '''

    no_spike = training_images == 0

    latency_images = 1.0 / training_images

    mask_latency_images = latency_images.masked_fill(no_spike, -torch.inf)

    min_val = torch.amin(latency_images, dim=(1,2,3), keepdim=True)
    max_val = torch.amax(mask_latency_images, dim=(1,2,3), keepdim=True)

    # Normalization will break if min and max are the same, one pixel in the image or all pixels of the same intensity
    # This is probably a situation you'll never run into with a dataset of images, but just a thought

    norm_latency_images = (latency_images - min_val) / (max_val - min_val) * (num_steps - 1)
    norm_latency_images.masked_fill_(no_spike, num_steps - 1)

    norm_latency_images = norm_latency_images.int()

    return norm_latency_images

if __name__ == '__main__':
    training_set = torchvision.datasets.FashionMNIST(root="assets/data/", train=True, transform=transforms.ToTensor(), download=True)

    training_images = training_set.data.unsqueeze(1)
    training_labels = training_set.targets

    norm_latency_images = latency(trainingDoG_test_tensor)
    print(norm_latency_images.size())
    print(norm_latency_images)

    norm_latency_images = norm_latency_images.flatten(start_dim=1)
    print(norm_latency_images.size())


    # training_images = training_set.data
    # training_labels = training_set.targets
    #
    # print(type(training_images))
    # print(training_images.unsqueeze(1).size())
    # print(type(training_labels))
    # print(training_labels.size())
    # train-images-idx3-ubyte, 60,000 training images, 26 Mbytes

    # 4) Implement class to take in the latency images