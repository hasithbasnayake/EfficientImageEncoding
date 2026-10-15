import torch

def lgn_preprocessing(dataset, shape, ctr, sur, threshold=1e-6):

    on_kernel, off_kernel = [kernel.unsqueeze(0).unsqueeze(1) for kernel in gen_lgn_receptive_fields(shape, ctr, sur)]

    on_dataset = torch.conv2d(dataset, on_kernel, padding='same')
    off_dataset = torch.conv2d(dataset, off_kernel, padding='same')

    on_dataset = torch.clamp(on_dataset, 0, 1)
    off_dataset = torch.clamp(off_dataset, 0, 1)

    lgn_dataset = torch.cat([on_dataset, off_dataset], dim=1)
    lgn_dataset[lgn_dataset < threshold] = 0

    return lgn_dataset

def gen_lgn_receptive_fields(shape, ctr, sur):
    ctr_kernel = gen_gaussian_kernel(shape, ctr)
    sur_kernel = gen_gaussian_kernel(shape, sur)

    on_kernel = ctr_kernel - sur_kernel
    off_kernel = -ctr_kernel + sur_kernel

    on_kernel = on_kernel - torch.mean(on_kernel)
    off_kernel = off_kernel - torch.mean(off_kernel)

    on_kernel = on_kernel / (torch.sum(torch.abs(on_kernel)) / 2 )
    off_kernel = off_kernel / (torch.sum(torch.abs(off_kernel)) / 2 )

    return on_kernel, off_kernel

def gen_gaussian_kernel(shape, sigma):
    h, w = shape
    row_dist = torch.linspace(-((h - 1.) / 2.), ((h - 1.) / 2.), h, dtype=torch.float32)
    col_dist = torch.linspace(-((w - 1.) / 2.), ((w - 1.) / 2.), w, dtype=torch.float32)

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

