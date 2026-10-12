# Define wta functions
import torch
import random

def update_weights(model_weights, input_spikes, spk_rec, wta_num, params):

    A_plus, A_minus, tau, mu_plus, mu_minus = params

    #spk_rec is 255, 100

    num_to_update, rnd_selection, first_idx = wta(spk_rec, wta_num)
    # so we need to collect the spike trains of the chosen and save it to out spike
    out_spike = spk_rec[:, num_to_update] # problem is this could be multiple
    # then we get the first index of this spike train, where this neuron spiked first
    # if there's actually an index of first spike (that's the if statement), get that index
    # then set all the other spikes after it equal to 0 in spk_rec
    # grab the weight matrix for the spiked neuron

    # then grab the spikes of the input neurons
    # and the times in which they spiked, not sure what format these are in

    # get the spikes of the output neuron again? not sure if I already got them

    # and then update the weights using masking

    # need to scale across the number of selections and provide weight updates
    # for each of those

    # output of this function is either 1 neuron or multiple neurons


def wta(spk_rec, wta_num):

    # print(spk_rec)
    # print(spk_rec.shape)
    # print(spk_rec.any(dim=1))
    # print(spk_rec.any(dim=1).sum().item())

    first_idx = torch.argmax(spk_rec.any(dim=1).int())
    # Need to get the edge case in which there's no spikes, we can do spk_rec.any to catch for that
    # Need to figure out if nothing spikes, when we randomly get 0, pass in an empty tensor of 255, 100 to test
    col_first_spk = spk_rec[first_idx, :]
    idx_of_neurons = (col_first_spk==1).nonzero().flatten()

    num_to_select = min(wta_num, len(idx_of_neurons))
    rnd_selection = random.sample(list(idx_of_neurons), num_to_select)

    print(col_first_spk.shape)
    print(col_first_spk)
    print(int(first_idx))
    print(idx_of_neurons)

    print(rnd_selection)


    return num_to_select, rnd_selection, first_idx