
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2
from skimage.transform import resize
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# Parámetros
image_size = 64
padded_size = 128
offset = (padded_size - image_size) // 2
psf_sigma = 1.5
steps_field = 20000
T0_field = 1.0
tau_field = steps_field / 5.0
lambda_smooth = 0.2
mu_entropy = 0.01
epsilon_rho = 0.1
nm_per_pixel = 33.3
um_per_pixel = nm_per_pixel / 1000
np.random.seed(42)

# Funciones auxiliares
def generate_psf_kernel(sigma):
    size = int(6 * sigma)
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def pad_kernel_to_image(kernel, shape):
    padded = np.zeros(shape, dtype=np.float32)
    k_shape = kernel.shape
    insert_slice = tuple(slice((s - ks) // 2, (s - ks) // 2 + ks) for s, ks in zip(shape, k_shape))
    padded[insert_slice] = kernel
    return padded

# Cargar imagen de célula
img_path = "confocal microscopy image.png"
image = Image.open(img_path).convert("L")
image_np = np.array(image, dtype=np.float32)
image_np /= image_np.max()
true_image = resize(image_np, (image_size, image_size), mode='reflect', anti_aliasing=True)

# Preparar padding
image_padded = np.zeros((padded_size, padded_size), dtype=np.float32)
image_padded[offset:offset + image_size, offset:offset + image_size] = true_image

# PSF y FFT
psf_kernel = generate_psf_kernel(psf_sigma)
kernel_padded = pad_kernel_to_image(psf_kernel, (padded_size, padded_size))
kernel_fft_padded = fft2(kernel_padded)

# Inicializar campo
rho_padded = np.abs(np.random.rand(padded_size, padded_size)) * 0.1

# Evolución Langevin
for step in range(steps_field):
    T_t = T0_field * np.exp(-step / tau_field)
    recon = np.real(ifft2(fft2(rho_padded) * kernel_fft_padded))
    residual = recon - image_padded
    d_chi2 = 2 * np.real(ifft2(fft2(residual) * kernel_fft_padded))
    laplacian = -4 * rho_padded + np.roll(rho_padded, 1, axis=0) + np.roll(rho_padded, -1, axis=0) +                 np.roll(rho_padded, 1, axis=1) + np.roll(rho_padded, -1, axis=1)
    d_smooth = -2 * lambda_smooth * laplacian
    d_entropy = mu_entropy * (1 + np.log(rho_padded + 1e-8))
    delta_S = d_chi2 + d_smooth + d_entropy
    noise = np.random.normal(0, 1, rho_padded.shape)
    rho_padded -= epsilon_rho * delta_S + np.sqrt(2 * epsilon_rho * T_t) * noise
    rho_padded = np.clip(rho_padded, 0, None)

# Campo continuo y reconstrucción
final_rho = rho_padded[offset:offset + image_size, offset:offset + image_size]
final_recon = np.real(ifft2(fft2(rho_padded) * kernel_fft_padded))[offset:offset + image_size, offset:offset + image_size]

# Producto campo × imagen reconstruida
product_rho_recon = final_rho * final_recon

# Suavizado + umbral
smoothed = gaussian_filter(product_rho_recon, sigma=0.8)
threshold_mask = smoothed > (0.2 * smoothed.max())

# Esqueletización
skeleton = skeletonize(threshold_mask)
labeled_skeleton = label(skeleton, connectivity=2)
regions = regionprops(labeled_skeleton)
lengths_um = [r.area * um_per_pixel for r in regions]

# Mostrar reconstrucción final
plt.figure(figsize=(6, 6))
plt.imshow(final_recon, cmap='hot')
plt.title("Imagen reconstruida (sin multiplicar)")
plt.axis('off')
plt.tight_layout()
plt.show()
