# %% [markdown]
# # Image Compression using Singular Value Decomposition (SVD)
# **Course:** Engineering Mathematics Foundation (B.Tech CSE - 1st Year)
# **Formula:** $A = U \Sigma V^T$

# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. LOAD IMAGE
# Place your photo in the same folder as this script and name it 'photo.jpg'
# Alternatively, use the synthetic pattern if you don't have a photo ready yet.
try:
    img_pil = Image.open('photo.jpg').convert('L')
    image = np.array(img_pil, dtype=float)
    print("Loaded 'photo.jpg' successfully.")
except FileNotFoundError:
    print("'photo.jpg' not found. Generating a synthetic test pattern instead.")
    x = np.linspace(-5, 5, 512)
    y = np.linspace(-5, 5, 512)
    X, Y = np.meshgrid(x, y)
    image = np.sin(X**2 + Y**2) + np.cos(2 * X) * np.sin(2 * Y)

print("Image Matrix Shape:", image.shape)

# Display Original Image
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.title("Original Image Matrix")
plt.axis('off')
plt.show()

# %% [markdown]
# ## Step 2: Compute Singular Value Decomposition (SVD)
# Decomposing matrix $A$ into $U$, $S$ (singular values), and $V^T$.

# %%
U, S, Vt = np.linalg.svd(image)

print("U shape:", U.shape)
print("Singular values (S) shape:", S.shape)
print("Vt shape:", Vt.shape)

# %% [markdown]
# ## Step 3: Reconstruct Compressed Image with Top $k$ Features

# %%
k = 30  # Try changing this to 10, 50, or 100

# Reconstruct image matrix using top k singular values
compressed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

# Calculate storage saved
original_size = image.size
compressed_size = U[:, :k].size + S[:k].size + Vt[:k, :].size
compression_ratio = (compressed_size / original_size) * 100

print(f"Keeping top {k} singular values.")
print(f"Storage used: {compression_ratio:.2f}% of the original size.")

# Display Compressed Image
plt.figure(figsize=(5, 5))
plt.imshow(compressed_image, cmap='gray')
plt.title(f"Compressed Image (k = {k})")
plt.axis('off')
plt.show()

# %% [markdown]
# ## Step 4: Multi-Panel Comparison ($k = 10, 30, 60$)

# %%
k_list = [10, 30, 60]
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original\n(100% storage)")
plt.axis('off')

for i, k_val in enumerate(k_list):
    approx = np.dot(U[:, :k_val], np.dot(np.diag(S[:k_val]), Vt[:k_val, :]))
    storage_pct = ((U[:, :k_val].size + S[:k_val].size + Vt[:k_val, :].size) / image.size) * 100
    
    plt.subplot(1, 4, i + 2)
    plt.imshow(approx, cmap='gray')
    plt.title(f"k = {k_val}\n({storage_pct:.1f}% storage)")
    plt.axis('off')

plt.tight_layout()
plt.show()