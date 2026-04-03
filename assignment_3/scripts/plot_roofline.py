import numpy as np
import matplotlib.pyplot as plt

# ---- A100 specs ----
peak_compute = 19500   # GFLOPs/s
bandwidth = 1600       # GB/s

# ---- AI range ----
ai = np.logspace(-2, 2, 200)

# Roofline
roofline = np.minimum(peak_compute, ai * bandwidth)

# ---- Your kernels ----
kernels = {
    "Gaussian": 19.512195122,
    "Sobel": 1.9,
    "Histogram": 0.5,
    "Equalisation": 0.66,
}

# ---- Plot ----
plt.figure()

plt.loglog(ai, roofline, label="Roofline (A100)")

# Ridge point
ridge_ai = peak_compute / bandwidth
plt.axvline(ridge_ai, linestyle='--')
plt.text(ridge_ai, peak_compute/2, "Ridge Point (12.1875)")

# Kernel points
for name, x in kernels.items():
    y = min(peak_compute, x * bandwidth)
    plt.scatter(x, y)
    plt.text(x, y, name)

plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (GFLOPs/s)")
plt.title("Roofline (A100)")

plt.grid(True, which="both")
plt.legend()

plt.show()