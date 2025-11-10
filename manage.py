import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# -----------------------------------
# Step 1: Load Image
# -----------------------------------
img = cv2.imread("sample_underwater.jpg")
if img is None:
    print("❌ Error: Image not found.")
    exit()

img = cv2.resize(img, (800, 600))

# -----------------------------------
# Step 2: Auto Gray-World Color Balance
# -----------------------------------
def gray_world_correction(image):
    b, g, r = cv2.split(image)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3
    b = np.clip(b * (mean_gray / mean_b), 0, 255)
    g = np.clip(g * (mean_gray / mean_g), 0, 255)
    r = np.clip(r * (mean_gray / mean_r), 0, 255)
    return cv2.merge((b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)))

img_corrected = gray_world_correction(img)

# -----------------------------------
# Step 3: Underwater Haze Reduction (Dark Channel Prior)
# -----------------------------------
def dark_channel(image, size=15):
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_img, kernel)

def estimate_atmosphere(image, dark):
    h, w = image.shape[:2]
    num_pixels = h * w // 1000
    dark_vec = dark.reshape(h * w)
    image_vec = image.reshape(h * w, 3)
    indices = dark_vec.argsort()[-num_pixels:]
    atmo = np.mean(image_vec[indices], axis=0)
    return atmo

def recover_scene(image, atmosphere, omega=0.95, t_min=0.1):
    image = image.astype('float64')
    dark = dark_channel(image / 255)
    transmission = 1 - omega * dark / np.max(atmosphere)
    transmission = cv2.max(transmission, t_min)
    J = np.empty_like(image)
    for i in range(3):
        J[:, :, i] = (image[:, :, i] - atmosphere[i]) / transmission + atmosphere[i]
    return np.clip(J, 0, 255).astype(np.uint8)

dark = dark_channel(img_corrected)
A = estimate_atmosphere(img_corrected, dark)
dehazed_img = recover_scene(img_corrected, A)

# -----------------------------------
# Step 4: CLAHE + Gamma + Wavelet Fusion
# -----------------------------------
lab = cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l = clahe.apply(l)
lab_clahe = cv2.merge((l, a, b))
clahe_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Adaptive Gamma correction (based on brightness)
gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
brightness = np.mean(gray) / 255
gamma = np.clip(1.2 + (0.4 - brightness), 0.7, 1.8)
gamma_corrected = np.array(255 * (clahe_img / 255) ** (1 / gamma), dtype='uint8')

# Wavelet Detail Enhancement
coeffs = pywt.dwt2(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY), 'db1')
cA, (cH, cV, cD) = coeffs
details = np.clip((cH + cV + cD) * 2.5, 0, 255).astype(np.uint8)
details = cv2.resize(details, (gamma_corrected.shape[1], gamma_corrected.shape[0]))
wavelet_merge = cv2.addWeighted(gamma_corrected, 0.8, cv2.cvtColor(details, cv2.COLOR_GRAY2BGR), 0.2, 0)

# -----------------------------------
# Step 5: Dynamic Depth-Aware Fusion
# -----------------------------------
depth_map = cv2.Laplacian(cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
depth_norm = cv2.normalize(np.abs(depth_map), None, 0, 1, cv2.NORM_MINMAX)

# Depth controls enhancement intensity
alpha = np.clip(0.6 + 0.4 * depth_norm.mean(), 0.6, 1.0)
beta = np.clip(30 * (1 - depth_norm.mean()), 10, 50)

fusion = cv2.convertScaleAbs(wavelet_merge, alpha=alpha, beta=beta)
final_output = cv2.bilateralFilter(fusion, 9, 75, 75)

# -----------------------------------
# Step 6: Display and Save Results
# -----------------------------------
cv2.imshow("Original Underwater", img)
cv2.imshow("Gray-World Corrected", img_corrected)
cv2.imshow("Dehazed Image (DCP)", dehazed_img)
cv2.imshow("Final Enhanced Output", final_output)

cv2.imwrite("underwater_enhanced_day5.jpg", final_output)
cv2.imwrite("underwater_dehazed_day5.jpg", dehazed_img)

# Comparison Plot
comparison = np.hstack((img, dehazed_img, final_output))
cv2.imwrite("comparison_day5.jpg", comparison)

plt.figure(figsize=(10,4))
plt.title("Day 5: Color Histogram Comparison")
plt.hist(img.ravel(), bins=256, color='gray', alpha=0.5, label='Original')
plt.hist(final_output.ravel(), bins=256, color='blue', alpha=0.5, label='Enhanced')
plt.legend()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
