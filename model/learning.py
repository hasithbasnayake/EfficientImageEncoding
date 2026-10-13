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


def wta(spk_rec, wta_num, strict=False):

    if spk_rec.any(dim=1).nonzero().numel() == 0:
        return torch.zeros_like(spk_rec)

    first_spk_timestep = torch.argmax(spk_rec.any(dim=1).int())
    col_first_spk_timestep = spk_rec[first_spk_timestep, :]
    idxs_of_first_spk_neurons = (col_first_spk_timestep==1).nonzero().flatten()

    wta_num = min(wta_num, len(idxs_of_first_spk_neurons))

    rnd_selection = random.sample(list(idxs_of_first_spk_neurons), wta_num)

    output = torch.zeros_like(spk_rec)
    output[:, rnd_selection] = spk_rec[:, rnd_selection]

    if strict:
        output[first_spk_timestep + 1:, :] = 0

    return output