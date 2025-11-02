import cv2
import numpy as np

# Step 1: Load image
img = cv2.imread("sample_underwater.jpg")

if img is None:
    print("Error: Image not found.")
    exit()

# Resize for display (optional)
img = cv2.resize(img, (800, 600))

# Step 2: White Balance Correction (remove blue/green tint)
# Convert to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Apply CLAHE to the L-channel (improves contrast)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# Merge back and convert to BGR
lab_clahe = cv2.merge((l_clahe, a, b))
img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Step 3: White balance tweak (reduce blue, boost red)
b, g, r = cv2.split(img_clahe)
r = cv2.addWeighted(r, 1.4, r, 0, 0)
g = cv2.addWeighted(g, 1.1, g, 0, 0)
b = cv2.addWeighted(b, 0.9, b, 0, 0)
color_corrected = cv2.merge((b, g, r))

# Step 4: Gamma correction (to brighten darker areas naturally)
gamma = 1.2  # <1 darker, >1 brighter
gamma_correction = np.array(255 * (color_corrected / 255) ** (1 / gamma), dtype='uint8')

# Step 5: Optional — Dehazing-like sharpness boost using bilateral filter
sharp = cv2.bilateralFilter(gamma_correction, 9, 75, 75)

# Step 6: Show and save results
cv2.imshow("Original Underwater Image", img)
cv2.imshow("Enhanced Underwater Image", sharp)
cv2.imwrite("underwater_enhanced.jpg", sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()