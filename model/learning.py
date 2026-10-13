import torch
import random

def update_weights(model_weights, input_spikes, spk_rec, wta_num, strict, params):

    A_plus, A_minus, tau, mu_plus, mu_minus = params

    output_spk = wta(spk_rec, wta_num, strict)

    t = torch.arange(255)
    dt = (t[:, None] - t[None, :]).float()

    w_plus = torch.where(dt > 0, A_plus * torch.exp(-dt / tau), 0.0)
    w_minus = torch.where(dt <= 0, A_minus * torch.exp(-dt.abs() / tau), 0.0)

    ltp = post.T @ w_plus @ spk_rec
    ltd = post.T @ w_minus @ spk_rec

    weight_update = ltp * (1 - model_weights)**mu_plus * model_weights**mu_minus

    return weight_update


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

    print(output.shape)

    return output