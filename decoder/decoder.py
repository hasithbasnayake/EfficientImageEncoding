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

def train_decoder(model, num_steps, decoder_loader):

    x, y = [], []

    model.eval()

    for idx, (batch_spikes, batch_images, batch_label) in enumerate(decoder_loader):

        batch_spikes = torch.transpose(batch_spikes, 0, 1)

        with torch.no_grad():
            spk_rec, _ = model.forward(batch_spikes)

        val, idxs = spk_rec.max(dim=0)

        first_spk_rec = torch.where(val==0, num_steps, idxs)
        org_batch_img = torch.flatten(batch_images, start_dim=1)

        x.append(first_spk_rec)
        y.append(org_batch_img)

    x = torch.cat((x)).numpy()
    y = torch.cat((y)).numpy()

    decoder = LinearRegression()
    decoder.fit(x,y)

    return decoder
