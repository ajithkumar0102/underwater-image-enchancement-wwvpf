import cv2
import numpy as np
import pywt

# Step 1: Load image
img = cv2.imread("sample_underwater.jpg")

if img is None:
    print("Error: Image not found.")
    exit()

# Resize for uniform processing
img = cv2.resize(img, (800, 600))

# Step 2: White Balance + CLAHE (contrast enhancement)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)
lab_clahe = cv2.merge((l_clahe, a, b))
img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Step 3: White balance tweak (reduce blue, boost red)
b, g, r = cv2.split(img_clahe)
r = cv2.addWeighted(r, 1.4, r, 0, 0)
g = cv2.addWeighted(g, 1.1, g, 0, 0)
b = cv2.addWeighted(b, 0.9, b, 0, 0)
color_corrected = cv2.merge((b, g, r))

# Step 4: Gamma correction (brighten darker areas)
gamma = 1.2
gamma_correction = np.array(255 * (color_corrected / 255) ** (1 / gamma), dtype='uint8')

# Step 5: Wavelet-based detail enhancement
gray = cv2.cvtColor(gamma_correction, cv2.COLOR_BGR2GRAY)
coeffs = pywt.dwt2(gray, 'db1')
cA, (cH, cV, cD) = coeffs
detail = cH + cV + cD
detail_enhanced = np.clip(detail * 2.0, 0, 255).astype(np.uint8)
detail_resized = cv2.resize(detail_enhanced, (gray.shape[1], gray.shape[0]))
wavelet_merge = cv2.addWeighted(gray, 0.8, detail_resized, 0.2, 0)
wavelet_merge = cv2.cvtColor(wavelet_merge, cv2.COLOR_GRAY2BGR)

# 🆕 Step 6: Adaptive Weight Map Calculation (Brightness + Contrast)
gray_base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
gray_wavelet = cv2.cvtColor(wavelet_merge, cv2.COLOR_BGR2GRAY)

# Normalize all to [0,1]
gray_base = gray_base / 255.0
gray_clahe = gray_clahe / 255.0
gray_wavelet = gray_wavelet / 255.0

# Compute local contrast using Laplacian variance
contrast_base = cv2.Laplacian(gray_base, cv2.CV_64F).var()
contrast_clahe = cv2.Laplacian(gray_clahe, cv2.CV_64F).var()
contrast_wavelet = cv2.Laplacian(gray_wavelet, cv2.CV_64F).var()

# Brightness maps (mean intensity)
brightness_base = np.mean(gray_base)
brightness_clahe = np.mean(gray_clahe)
brightness_wavelet = np.mean(gray_wavelet)

# Combine contrast + brightness to create adaptive weights
w1 = 0.3 + 0.4 * contrast_clahe + 0.2 * brightness_clahe
w2 = 0.3 + 0.4 * contrast_wavelet + 0.2 * brightness_wavelet
w3 = 1.0 - (w1 + w2) / 2  # ensures normalized distribution

# Normalize weights to sum to 1
sum_w = w1 + w2 + w3
w1, w2, w3 = w1/sum_w, w2/sum_w, w3/sum_w

# 🧩 Step 7: Adaptive Weighted Fusion
adaptive_fused = cv2.addWeighted(img_clahe, w1, wavelet_merge, w2, 0)
adaptive_fused = cv2.addWeighted(adaptive_fused, 1.0, gamma_correction, w3, 0)

# Step 8: Final smooth sharpening
final_output = cv2.bilateralFilter(adaptive_fused, 9, 75, 75)

# Step 9: Display and save results
cv2.imshow("Original Underwater Image", img)
cv2.imshow("Adaptive Weighted Fusion Enhanced", final_output)
cv2.imwrite("underwater_enhanced_day3.jpg", final_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
