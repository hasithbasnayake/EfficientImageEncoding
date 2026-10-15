import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

# In the original code, the train loader is used and is initialized with train_set, batch_size=1, shuffle=True
# train_set is a subset of train_dataset which is the DoG transformed original dataset. These are float tensors
# Note that uniquely in the existing code, train_set contains 3 channels, the first two being the channels for
# actual 2-channel LGN images, and the 3rd being a hack that contains the original image

# When you create spike_set using the SpikeDataset class, note that the image in (spikes, image label) is actually an
# tensor of uint8 not a float, so we need to convert to a float and divide to 255.
# We should probably pull the float conversion out of lgn_preprocessing into the normal pipeline

def train_decoder(model, spike_set, num_steps, decoder_loader):

    x, y = [], []

    train_length = 1

    spikes, image, label = spike_set[0]
    print(type(spike_set[0]))
    latency_images = spike_set.latency_images

    print(f"type of image: {image.dtype}")
    print(f"shape of image: {image.shape}")
    print(f"type of spike: {spikes.dtype}")
    print(f"shape of spike: {spikes.shape}")
    print(f"type of latency: {latency_images.dtype}")
    print(f"shape of latency: {latency_images.shape}")

    test_spike, test_image, test_label = next(iter(decoder_loader))
    test = next(iter(decoder_loader))
    print(type(test[0]))
    print(len(test))
    print(test[0].shape)
    print(test_spike.shape)
    print(test[1].shape)
    print(test[2].shape)
    print(f" Type of test: {type(test_spike)}")
    print(len(test_label))
    # print(test)

    model.eval()

    for i in range(train_length):

        batch_spikes, batch_images, batch_label = next(iter(decoder_loader))

    # for idx, (batch_spikes, batch_images, batch_label) in enumerate(DataLoader):

        batch_spikes = torch.transpose(batch_spikes, 0, 1)

        with torch.no_grad():
            spk_rec, mem_rec = model.forward(batch_spikes)

        val, idxs = spk_rec.max(dim=0)

        first_spk_rec = torch.where(val==0, num_steps, idxs)
        org_batch_img = torch.flatten(batch_images, start_dim=1)

        x.append(first_spk_rec)
        y.append(org_batch_img)

    x = torch.cat((x))
    y = torch.cat((y))
    print(x.shape)
    print(y.shape)

    x, y = [],[]

    for i in range(64):
        spikes, image, label = spike_set[i]

        print(f"Shape of spikes at {i}: {spikes.shape}")

        with torch.no_grad():
            spk_rec, mem_rec = model.forward(spikes.squeeze())

        vals, idxs = spk_rec.max(dim=0)

        print(f"Shape of vals at {i}: {vals.shape}")
        print(f"Shape of idxs at {i}: {idxs.shape}")

        first_spk_rec = torch.where(vals == 0, num_steps, idxs)
        print(f"Shape of first_spk_rec at {i}: {first_spk_rec.shape}")

        x.append(first_spk_rec)
        y.append(image.squeeze().view(-1))
        print(f"Shape of image: {image.squeeze().view(-1).shape}")

    print(f"Shape of x: {len(x)}")
    print(f"Shape of y: {len(y)}")
    x = torch.stack(x).numpy()
    y = torch.stack(y).numpy()
    print(f"Shape of x: {x.shape}")
    print(f"Shape of y: {y.shape}")

    # print(f"Test image shape: {torch.flatten(test_image, start_dim=1).shape}")
    #
    # test_spike = torch.transpose(test_spike, 0, 1)
    # print(f"Test spike after transforming: {test_spike.shape}")
    #
    # with torch.no_grad():
    #     spk_rec, mem_rec = model.forward(test_spike)
    #
    # print(f"Shape of spk_rec: {spk_rec.shape}")
    # vals, idxs = spk_rec.max(dim=0)
    # print(f"Shape of vals: {vals.shape}")
    # first_spk_rec = torch.where(vals == 0, num_steps, idxs)
    # print(f"Shape of first_spk_rec at: {first_spk_rec.shape}")

    #
    # decoder = LinearRegression()
    # decoder.fit(x,y)

    # return decoder
