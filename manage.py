import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn

# -------------------------------------------------------
# 1️⃣  Minimal CNN Model (acts as placeholder)
# -------------------------------------------------------
class TinyEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))

def ai_enhance(img):
    model = TinyEnhancer()
    model_path = "underwater_ai_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    else:
        print("⚠️  No pretrained model found – using Day 5 fallback.")

    # Pre-process
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    inp = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).float()

    with torch.no_grad():
        out = model(inp).squeeze(0).permute(1,2,0).numpy()

    out = np.clip(out*255,0,255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# -------------------------------------------------------
# 2️⃣  Classical Enhancement Fallback (Day 5 condensed)
# -------------------------------------------------------
def classical_enhance(img):
    # Gray-world balance
    b,g,r = cv2.split(img)
    mb,mg,mr = np.mean(b),np.mean(g),np.mean(r)
    mgm = (mb+mg+mr)/3
    img = cv2.merge((
        np.clip(b*mgm/mb,0,255),
        np.clip(g*mgm/mg,0,255),
        np.clip(r*mgm/mr,0,255)
    )).astype(np.uint8)

    # CLAHE + gamma
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(3.0,(8,8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
    gamma = 1.2
    img = np.array(255*(img/255)**(1/gamma),dtype='uint8')

    # Wavelet detail
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cA,(cH,cV,cD)=pywt.dwt2(gray,'db1')
    d=np.clip((cH+cV+cD)*2,0,255).astype('uint8')
    d=cv2.cvtColor(cv2.resize(d,(img.shape[1],img.shape[0])),cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img,0.85,d,0.15,0)

# -------------------------------------------------------
# 3️⃣  Load + Process
# -------------------------------------------------------
import os
img = cv2.imread("sample_underwater.jpg")
if img is None:
    print("Image not found"); exit()
img = cv2.resize(img,(800,600))

try:
    final_ai = ai_enhance(img)
except Exception as e:
    print("AI enhancement failed:", e)
    final_ai = classical_enhance(img)

cv2.imshow("Original", img)
cv2.imshow("AI / Classical Enhanced", final_ai)
cv2.imwrite("underwater_day6_ai.jpg", final_ai)
cv2.waitKey(0)
cv2.destroyAllWindows()
