def latency(images: torch.Tensor, num_steps: int = 255):
    """
    Convert pixel intensities to spike latencies using a 1 / x encoding.

    Parameters
    ----------
    images : Torch.tensor
        An image tensor of the shape [N, C, H, W] where N is the number of images,
        C is the number of channels, H is the height, and W is the width.
    num_steps : int
        The number of timesteps the spike latencies should be normalized across

    Returns
    ----------
    torch.Tensor
        An Int tensor of the shape [N, C, H, W] with values in the range of [0, num_steps - 1].
        Zero-intensity pixels fire at the last step (num_steps - 1).

    """

    # Create a boolean mask where the pixel val is 0
    bool_mask_no_spike = images == 0

    # Latency encoding of images
    latency_images = 1.0 / images

    masked_latency_images = latency_images.masked_fill(bool_mask_no_spike, -torch.inf)

    min_val = torch.amin(latency_images, dim=(1,2,3), keepdim=True)
    max_val = torch.amax(masked_latency_images, dim=(1,2,3), keepdim=True)

    # NOTE: Normalization will break if min and max are the same value
    # (cont.) this will happen when an image has a single light pixel or all pixels are the same value

    norm_latency_images = (latency_images - min_val) / (max_val - min_val) * (num_steps - 1)
    norm_latency_images.masked_fill_(bool_mask_no_spike, num_steps - 1)

    norm_latency_images = norm_latency_images.int()

    return norm_latency_images