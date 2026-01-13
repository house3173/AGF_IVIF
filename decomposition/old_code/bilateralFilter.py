import numpy as np
import scipy.ndimage
import warnings

def bilateral_filter(data, edge=None, edge_min=None, edge_max=None, 
                     sigma_spatial=None, sigma_range=None, 
                     sampling_spatial=None, sampling_range=None):
    """
    Bilateral and Cross-Bilateral Filter using the Bilateral Grid.
    Ported from MATLAB code by Jiawen (Kevin) Chen.
    """
    
    # --- Input Validation and Setup ---
    data = np.asarray(data, dtype=np.float64)
    
    if data.ndim > 2:
        raise ValueError('data must be a greyscale image with size [height, width]')
        
    if edge is None:
        edge = data
    else:
        edge = np.asarray(edge, dtype=np.float64)
        
    if edge.ndim > 2:
        raise ValueError('edge must be a greyscale image with size [height, width]')
        
    if data.shape != edge.shape:
        raise ValueError('data and edge must be of the same size')

    input_height, input_width = data.shape

    # --- Parameter Defaults ---
    if edge_min is None:
        edge_min = edge.min()
        warnings.warn(f'edge_min not set! Defaulting to: {edge_min}')
        
    if edge_max is None:
        edge_max = edge.max()
        warnings.warn(f'edge_max not set! Defaulting to: {edge_max}')
        
    edge_delta = edge_max - edge_min
    
    if sigma_spatial is None:
        sigma_spatial = min(input_width, input_height) / 16.0
        
    if sigma_range is None:
        sigma_range = 0.1 * edge_delta
        
    if sampling_spatial is None:
        sampling_spatial = sigma_spatial
        
    if sampling_range is None:
        sampling_range = sigma_range

    # --- Grid Parameters ---
    derived_sigma_spatial = sigma_spatial / sampling_spatial
    derived_sigma_range = sigma_range / sampling_range

    padding_xy = int(np.floor(2 * derived_sigma_spatial)) + 1
    padding_z = int(np.floor(2 * derived_sigma_range)) + 1

    # Allocate 3D grid
    # Note: Python uses 0-based indexing, so size calculations are slightly adjusted from MATLAB
    downsampled_width = int(np.floor((input_width - 1) / sampling_spatial)) + 1 + 2 * padding_xy
    downsampled_height = int(np.floor((input_height - 1) / sampling_spatial)) + 1 + 2 * padding_xy
    downsampled_depth = int(np.floor(edge_delta / sampling_range)) + 1 + 2 * padding_z

    grid_data = np.zeros((downsampled_height, downsampled_width, downsampled_depth))
    grid_weights = np.zeros((downsampled_height, downsampled_width, downsampled_depth))

    # --- Compute Downsampled Indices ---
    # meshgrid in numpy 'ij' indexing matches matrix logic (row, col)
    ii, jj = np.meshgrid(np.arange(input_height), np.arange(input_width), indexing='ij')

    # Calculate indices (coordinates in the grid)
    # Python round() rounds to nearest even number for .5, but MATLAB rounds away from zero.
    # We use np.round() which behaves like MATLAB's round (mostly) but we need to cast to int.
    # Also removed the +1 that MATLAB uses for 1-based indexing.
    di = np.round(ii / sampling_spatial).astype(int) + padding_xy
    dj = np.round(jj / sampling_spatial).astype(int) + padding_xy
    dz = np.round((edge - edge_min) / sampling_range).astype(int) + padding_z

    # --- Perform Scatter (Vectorized) ---
    # Instead of the slow MATLAB loop, we use np.add.at for fast accumulation
    
    # Flatten arrays to list of coordinates
    flat_di = di.ravel()
    flat_dj = dj.ravel()
    flat_dz = dz.ravel()
    flat_data = data.ravel()
    
    # Handle NaN values in data (mask them out)
    valid_mask = ~np.isnan(flat_data)
    
    valid_di = flat_di[valid_mask]
    valid_dj = flat_dj[valid_mask]
    valid_dz = flat_dz[valid_mask]
    valid_data = flat_data[valid_mask]
    
    # Accumulate data
    np.add.at(grid_data, (valid_di, valid_dj, valid_dz), valid_data)
    # Accumulate weights (count)
    np.add.at(grid_weights, (valid_di, valid_dj, valid_dz), 1)

    # --- Make Gaussian Kernel ---
    kernel_width = 2 * derived_sigma_spatial + 1
    kernel_height = kernel_width
    kernel_depth = 2 * derived_sigma_range + 1

    half_kernel_width = np.floor(kernel_width / 2)
    half_kernel_height = np.floor(kernel_height / 2)
    half_kernel_depth = np.floor(kernel_depth / 2)

    # Grid for kernel
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(kernel_width), 
        np.arange(kernel_height), 
        np.arange(kernel_depth), 
        indexing='xy' # Use xy to match MATLAB's meshgrid behavior for kernel construction
    )
    
    # Adjust coordinates
    grid_x -= half_kernel_width
    grid_y -= half_kernel_height
    grid_z -= half_kernel_depth
    
    # Compute squared radius
    grid_r_squared = ((grid_x**2 + grid_y**2) / (derived_sigma_spatial**2)) + \
                     ((grid_z**2) / (derived_sigma_range**2))
    
    kernel = np.exp(-0.5 * grid_r_squared)

    # --- Convolve ---
    # scipy.ndimage.convolve is equivalent to MATLAB's convn with 'same' via mode='constant'
    blurred_grid_data = scipy.ndimage.convolve(grid_data, kernel, mode='constant', cval=0.0)
    blurred_grid_weights = scipy.ndimage.convolve(grid_weights, kernel, mode='constant', cval=0.0)

    # --- Divide ---
    # Avoid divide by zero
    mask_zero = (blurred_grid_weights == 0)
    blurred_grid_weights[mask_zero] = -2 # Safe value, won't be read
    
    normalized_blurred_grid = blurred_grid_data / blurred_grid_weights
    normalized_blurred_grid[blurred_grid_weights < -1] = 0 # Handle undefined regions

    # --- Upsample (Interpolation) ---
    # Recalculate coordinates (floating point this time, no rounding)
    # Note: ii and jj were created with indexing='ij' earlier
    di_query = (ii / sampling_spatial) + padding_xy
    dj_query = (jj / sampling_spatial) + padding_xy
    dz_query = (edge - edge_min) / sampling_range + padding_z

    # Stack coordinates for map_coordinates: shape (3, num_pixels)
    coords = np.vstack((di_query.ravel(), dj_query.ravel(), dz_query.ravel()))

    # Trilinear interpolation
    # map_coordinates is extremely efficient for this. 
    # order=1 matches MATLAB's interpn default (linear).
    output_flat = scipy.ndimage.map_coordinates(normalized_blurred_grid, coords, order=1, mode='nearest')
    
    output = output_flat.reshape(input_height, input_width)

    return output

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy image or load one using cv2/PIL
    # Creating a simple gradient image with noise
    img = np.zeros((100, 100))
    for i in range(100):
        img[i, :] = i / 100.0
    
    # Add noise
    noisy_img = img + 0.05 * np.random.randn(100, 100)
    noisy_img = np.clip(noisy_img, 0, 1)

    # Run filter
    print("Running Bilateral Filter...")
    result = bilateral_filter(noisy_img, sigma_spatial=4, sigma_range=0.1)
    print("Done.")
    print(f"Input shape: {noisy_img.shape}, Output shape: {result.shape}")