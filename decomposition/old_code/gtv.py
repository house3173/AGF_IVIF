import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from decomposition.old_code.gaussFilter import gauss_filter

def gtv(I, lambd=0.005, sigma12=0.0004, sigma2=0.004, N=4, flog=1):
    """
    Gradient Total Variation (GTV) Filter implemented in Python.
    Ported from MATLAB code.
    """
    # Convert image to double (float [0, 1])
    if I.dtype != np.float64 and I.dtype != np.float32:
        I = I.astype(np.float64) / 255.0
    else:
        I = I.astype(np.float64)

    epsilon = 0.001
    S = I.copy()

    for i in range(N):
        wx, wy = compute_weights(S, sigma12, sigma2, epsilon, flog)
        S = solve_linear_equation(I, wx, wy, lambd)
        print(f"Iteration {i+1}/{N} completed.")
        
    return S

def compute_weights(s, sigma12, sigma2, epsilon, flog):
    # --- Compute original gradients ---
    # diff along axis 1 (columns/x) -> matches MATLAB diff(s, 1, 2)
    dxo = np.diff(s, axis=1)
    # Pad width ((top, bottom), (left, right), (channels...))
    # Pad 'post' (right side) with 0
    padding_x = ((0, 0), (0, 1)) if s.ndim == 2 else ((0, 0), (0, 1), (0, 0))
    dxo = np.pad(dxo, padding_x, mode='constant', constant_values=0)

    # diff along axis 0 (rows/y) -> matches MATLAB diff(s, 1, 1)
    dyo = np.diff(s, axis=0)
    padding_y = ((0, 1), (0, 0)) if s.ndim == 2 else ((0, 1), (0, 0), (0, 0))
    dyo = np.pad(dyo, padding_y, mode='constant', constant_values=0)

    # Compute weights wxo, wyo
    # max over axis 2 (channels) if 3D, else just abs
    if s.ndim == 3:
        grad_mag_x = np.max(np.abs(dxo), axis=2)
        grad_mag_y = np.max(np.abs(dyo), axis=2)
    else:
        grad_mag_x = np.abs(dxo)
        grad_mag_y = np.abs(dyo)

    wxo = np.maximum(grad_mag_x, epsilon) ** (-1)
    wyo = np.maximum(grad_mag_y, epsilon) ** (-1)

    if flog == 1:
        si = gauss_filter(s, sigma2)
        
        dxs = np.diff(si, axis=1)
        dxs = np.pad(dxs, padding_x, mode='constant', constant_values=0)
        
        dys = np.diff(si, axis=0)
        dys = np.pad(dys, padding_y, mode='constant', constant_values=0)

        # Logic matches MATLAB exactly, but note:
        # MATLAB code: wxt=min(max((exp((dxs.*dxs)./(2*sigma12))),[],3),epsilon).^(-1);
        # If exp(...) >= 1 and epsilon=0.001, min(..., epsilon) is always epsilon.
        # This effectively makes wxt constant (1/epsilon). 
        # I strictly followed the provided MATLAB code here.
        
        term_x = np.exp((dxs * dxs) / (2 * sigma12))
        term_y = np.exp((dys * dys) / (2 * sigma12))

        if s.ndim == 3:
            term_x = np.max(term_x, axis=2)
            term_y = np.max(term_y, axis=2)

        # MATLAB: min(..., epsilon)
        wxt = np.minimum(term_x, epsilon) ** (-1)
        wyt = np.minimum(term_y, epsilon) ** (-1)
        
        wxt = wxo * wxt
        wyt = wyo * wyt
    else:
        wxt = wxo
        wyt = wyo

    # Zero out boundaries matches MATLAB: wxt(:,end)=0; wyt(end,:)=0;
    wxt[:, -1] = 0
    wyt[-1, :] = 0

    return wxt, wyt

def solve_linear_equation(in_img, wx, wy, lambd):
    h, w = in_img.shape[:2]
    c = in_img.shape[2] if in_img.ndim == 3 else 1
    n = h * w

    # Flatten weights using Fortran order ('F') to match MATLAB column-major indexing
    wx_flat = wx.flatten(order='F')
    wy_flat = wy.flatten(order='F')

    # Construct ux, uy (padded shifted versions)
    ux_flat = np.concatenate([np.zeros(h), wx_flat[:-h]])
    uy_flat = np.concatenate([np.zeros(1), wy_flat[:-1]])

    D = wx_flat + ux_flat + wy_flat + uy_flat
    
    # Construct Sparse Matrix B
    offsets = np.array([-h, -1])
    data = np.vstack([-wx_flat, -wy_flat])
    B = sp.spdiags(data, offsets, n, n)
    
    # L = B + B' + D (diagonal)
    L = B + B.transpose() + sp.spdiags(D, 0, n, n)
    
    # A = I + lambda * L
    A = sp.eye(n) + lambd * L

    # Preconditioner using Incomplete LU
    try:
        spilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=2)
        M = spla.LinearOperator((n, n), matvec=spilu.solve)
    except:
        M = None

    out = in_img.copy()

    # Solve for each channel
    if c > 1:
        for i in range(c):
            tin = in_img[:, :, i].flatten(order='F')
            # SỬA LỖI TẠI ĐÂY: Thay tol=0.1 thành rtol=0.1
            tout, info = spla.cg(A, tin, rtol=0.1, maxiter=100, M=M)
            out[:, :, i] = tout.reshape((h, w), order='F')
    else:
        tin = in_img.flatten(order='F')
        # SỬA LỖI TẠI ĐÂY: Thay tol=0.1 thành rtol=0.1
        tout, info = spla.cg(A, tin, rtol=0.1, maxiter=100, M=M)
        out = tout.reshape((h, w), order='F')

    return out

# --- Example Usage ---
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # Create a dummy image or load one
    # img = cv2.imread('path_to_image.jpg')
    # For demo, creating a noisy synthetic image
    img = np.zeros((100, 100))
    img[20:80, 20:80] = 1.0
    noise = np.random.normal(0, 0.1, img.shape)
    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    
    # Stack to make it 3 channels (RGB) like a real image
    img_noisy_rgb = np.stack([img_noisy]*3, axis=2)

    print("Running GTV...")
    result = gtv(img_noisy_rgb, lambd=5, N=4)
    print("Done.")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Noisy Input")
    plt.imshow(img_noisy, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("GTV Output")
    plt.imshow(result[:,:,0], cmap='gray')
    plt.show()