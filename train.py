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

if __name__ == '__main__':
    pass