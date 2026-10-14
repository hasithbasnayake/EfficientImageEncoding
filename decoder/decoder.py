import torch
from sklearn.linear_model import LinearRegression

def train_decoder(model, test_spike_set, num_steps):
    X = []
    Y = []

    train_length = len(test_spike_set)

    for i in range(train_length):
        spikes, image, label = test_spike_set[i]

        with torch.no_grad():
            spk_rec, mem_rec = model.forward(spikes.squeeze())

        vals, idxs = spk_rec.max(dim=0)

        first_spk_rec = torch.where(vals == 0, num_steps, idxs)

        X.append(first_spk_rec)
        Y.append(image.squeeze().view(-1))

    X = torch.stack(X).detach().numpy()
    Y = torch.stack(Y).detach().numpy()

    decoder = LinearRegression()
    decoder.fit(X,Y)

    return decoder
