import torch
import numpy as np
import matplotlib.pyplot as plt

# def difference_of_gaussians(dim, ctr, sur):
#
#     kernel = gen_gaussian_kernel(dim, sur)
#     lgn_field_on, lgn_field_off = gen_receptive_fields(kernel)
#
#     # Convolve with dataset here
#
#     return None
#
# def gen_receptive_fields(kernel):
#     pass

def gen_gaussian_kernel(shape, sigma):
    h, w = shape
    row_dist = torch.linspace(-((h - 1.) / 2.), ((h - 1.) / 2.), h, dtype=torch.float64)
    col_dist = torch.linspace(-((w - 1.) / 2.), ((w - 1.) / 2.), w, dtype=torch.float64)

    dist_matrix = torch.square(row_dist[:, None]) + torch.square(col_dist[None, :])

    g_kernel = torch.exp(-dist_matrix / (2. * sigma**2))
    g_kernel[g_kernel < torch.finfo(g_kernel.dtype).eps * g_kernel.max()] = 0
    g_kernel /= g_kernel.sum()

    return g_kernel

# def gen_gaussian_kernel(shape, sigma):
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y, x = np.ogrid[-m:m + 1, -n:n + 1]
#     h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     h /= h.sum()
#     return h

