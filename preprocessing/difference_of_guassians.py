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

def gen_gaussian_kernel_test(shape, sigma):
    x, y = shape
    x_dist = torch.linspace(-((x - 1.) / 2.), ((x - 1.) / 2.), x)
    y_dist = torch.linspace(-((y - 1.) / 2.), ((y - 1.) / 2.), y)

    print(x_dist)
    print(y_dist)
    print(f"x: {x}")
    print(f"y: {y}")

    pass

def gen_gaussian_kernel(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    # print(m, n)
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    # print(x)
    # print(x*x + y*y)
    # print(y*y)
    # print(y)
    # print((-(x*x + y*y) / (2. * sigma * sigma)))
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    bool_mask = h < np.finfo(h.dtype).eps * h.max()
    # print(bool_mask)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # print(h)
    h /= h.sum()
    return h

