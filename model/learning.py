# Define wta functions
import torch
import random

def update_weights(model, spk_rec, wta_num, params):

    num_to_update, rnd_selection = wta(spk_rec, wta_num)

    # need to scale across the number of selections and provide weight updates
    # for each of those


def wta(spk_rec, wta_num):

    # print(spk_rec)
    # print(spk_rec.shape)
    # print(spk_rec.any(dim=1))
    # print(spk_rec.any(dim=1).sum().item())

    first_idx = torch.argmax(spk_rec.any(dim=1).int())
    # Need to get the edge case in which there's no spikes, we can do spk_rec.any to catch for that
    col_first_spk = spk_rec[first_idx, :]
    idx_of_neurons = (col_first_spk==1).nonzero().squeeze()

    num_to_select = wta_num if wta_num <= len(idx_of_neurons) else len(idx_of_neurons)
    rnd_selection = random.sample(list(idx_of_neurons), num_to_select)

    print(col_first_spk.shape)
    print(col_first_spk)
    print(int(first_idx))
    print(idx_of_neurons)

    print(rnd_selection)


    return num_to_select, rnd_selection