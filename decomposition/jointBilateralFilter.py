import numpy as np
from numba import njit, prange

def joint_bilateral_filter(p, I, sigma_spatial, sigma_range):
    """
    Joint Bilateral Filter
    p: input image (float32, [0,1])
    I: guidance image (float32, [0,1])
    """
    radius = int(3 * sigma_spatial)
    kernel_size = 2 * radius + 1

    # Spatial Gaussian
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_spatial**2))

    H, W = p.shape
    padded_p = np.pad(p, radius, mode="reflect")
    padded_I = np.pad(I, radius, mode="reflect")

    output = np.zeros_like(p, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            patch_p = padded_p[i:i+kernel_size, j:j+kernel_size]
            patch_I = padded_I[i:i+kernel_size, j:j+kernel_size]

            range_kernel = np.exp(
                -((patch_I - padded_I[i+radius, j+radius])**2)
                / (2 * sigma_range**2)
            )

            weights = spatial_kernel * range_kernel
            weights /= np.sum(weights) + 1e-8

            output[i, j] = np.sum(weights * patch_p)

    return np.clip(output, 0, 1)

@njit(parallel=True, fastmath=True)
def joint_bilateral_filter_numba(
    p,
    I,
    sigma_spatial,
    sigma_range
):
    H, W = p.shape
    radius = int(3 * sigma_spatial)
    out = np.zeros_like(p)

    # Precompute spatial kernel
    size = 2 * radius + 1
    spatial = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dx = i - radius
            dy = j - radius
            spatial[i, j] = np.exp(-(dx*dx + dy*dy) / (2*sigma_spatial*sigma_spatial))

    for y in prange(H):
        for x in range(W):
            wsum = 0.0
            psum = 0.0
            Ic = I[y, x]

            for dy in range(-radius, radius + 1):
                yy = min(max(y + dy, 0), H - 1)
                for dx in range(-radius, radius + 1):
                    xx = min(max(x + dx, 0), W - 1)

                    gr = np.exp(
                        -((I[yy, xx] - Ic) ** 2) / (2 * sigma_range * sigma_range)
                    )
                    w = spatial[dy + radius, dx + radius] * gr

                    wsum += w
                    psum += w * p[yy, xx]

            out[y, x] = psum / (wsum + 1e-8)

    return out