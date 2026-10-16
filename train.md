First, pause, stop, resume don't become actions on the same training run, they should actually kill the run and restart intelligently. So rather than pause and resume, it becomes checkpoint and resume.

You can start a training run using: `python --start /config.json`

It reads config.json from the address you've given it and it imports the keys into a dataset class. It catches any errors if the keys are incorrect or the values are wrong. There should be some type enforcement here I'd imagine.

Once it reads the config.json, it creates a new folder with an optional name (another flag) + date, and time, creates a config.json file with the contents of the config.json you've given it and begins the training run. 

It begins the training run by passing in parameters provided by config.json into train_model. Metadata about the run stored in config, like torch seed values are provided into train_model for things like setting the random seed of the network as well as used above all the training functions to set the seed for everything (or maybe I can just set the seed once at the top)

After training, the run automatically goes into training the decoder using train_decoder, and then testing the decoder on a held out test set (test set will be obtained from FashionMNIST test).

Note that we use a signal handler when a user Ctrl+C's to set a flag which we check between images in train_model to cancel and benchmark.

train_model, train_decoder, and test_decoder use log.txt in the folder to set epochs and where they are in the training data and overall run. The python code uses this to determine where to start from when resuming.

`python --resume /run_folder --stages `
Given a path to a model folder, it checks the config for the parameters and the log.txt for where it left off. It resumes and will decode and test. 

`python --resume /run_folder --stages decoder,test`
Given a path to a model folder, but with the flags for decoder and test, it skips straight to decoding and testing. 

`python --resume /run_folder --stages train,decoder,test`
Given a path to a model folder, but with the flags for decoder and test, it will continue training into to decoding and testing. 

`python --resume /run_folder --stages train,decoder`
Given a path to a model folder, but with the flags for decoder and test, it will continue training but will only decode.

How can I continuing training a model past its number of epochs given in its config? 