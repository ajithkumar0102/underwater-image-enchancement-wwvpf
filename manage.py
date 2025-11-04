import cv2
import numpy as np
import pywt

# Step 1: Load image
img = cv2.imread("sample_underwater.jpg")

if img is None:
    print("Error: Image not found.")
    exit()

# Optional resize for consistency
img = cv2.resize(img, (800, 600))

# Step 2: White Balance + CLAHE (contrast enhancement)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

lab_clahe = cv2.merge((l_clahe, a, b))
img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Step 3: White balance tweak (reduce blue tint, enhance red)
b, g, r = cv2.split(img_clahe)
r = cv2.addWeighted(r, 1.4, r, 0, 0)
g = cv2.addWeighted(g, 1.1, g, 0, 0)
b = cv2.addWeighted(b, 0.9, b, 0, 0)
color_corrected = cv2.merge((b, g, r))

# Step 4: Gamma correction (brighten image naturally)
gamma = 1.2  # <1 darker, >1 brighter
gamma_correction = np.array(255 * (color_corrected / 255) ** (1 / gamma), dtype='uint8')

# Step 5: Wavelet-based detail enhancement (fixed version)
gray = cv2.cvtColor(gamma_correction, cv2.COLOR_BGR2GRAY)

# Apply Discrete Wavelet Transform
coeffs = pywt.dwt2(gray, 'db1')
cA, (cH, cV, cD) = coeffs

# Combine details and scale
detail = cH + cV + cD
detail_enhanced = np.clip(detail * 2.0, 0, 255).astype(np.uint8)

# Resize back to original gray size
detail_resized = cv2.resize(detail_enhanced, (gray.shape[1], gray.shape[0]))

# Merge wavelet detail with base gray image
wavelet_merge = cv2.addWeighted(gray, 0.8, detail_resized, 0.2, 0)
wavelet_merge = cv2.cvtColor(wavelet_merge, cv2.COLOR_GRAY2BGR)

# Step 6: Weighted Fusion (main improvement today)
# Combine CLAHE, color_corrected, and wavelet-enhanced images
fused = cv2.addWeighted(img_clahe, 0.3, color_corrected, 0.4, 0)
fused = cv2.addWeighted(fused, 1.0, wavelet_merge, 0.3, 0)

# Step 7: Final smoothing for crisp, noise-free result
final_output = cv2.bilateralFilter(fused, 9, 75, 75)

# Step 8: Display and save
cv2.imshow("Original Underwater Image", img)
cv2.imshow("Weighted Fusion Enhanced Image", final_output)
cv2.imwrite("underwater_enhanced_day2.jpg", final_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
