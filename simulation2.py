import cv2
import torch
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn

# ======================================================
# 1. CONFIG
# ======================================================

#IMG_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test\forest\20062.jpg"
IMG_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test\mountain\23318.jpg"
#IMG_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test\sea\20124.jpg"
#IMG_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test\street\20080.jpg"
#IMG_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test\glacier\22123.jpg"
#IMG_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test\buildings\20228.jpg"

CLASS_NAMES = ['buildings','forest','glacier','mountain','sea','street']

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

# ======================================================
# 2. MODEL (IDENTIK DENGAN TRAINING)
# ======================================================

class AdaptiveCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4,4))
        )

        self.classifier = nn.Sequential(

            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,6)

        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.classifier(x)

# Load trained weights
model = AdaptiveCNN().to(DEVICE)
model.load_state_dict(
    torch.load("best_model_cpu.pth", map_location=DEVICE, weights_only=True)
)
model.eval()

# ======================================================
# 3. TRANSFORM
# ======================================================

def get_transform(res):
    return transforms.Compose([
        transforms.Resize((res,res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

# ======================================================
# 4. PREDICTION (REAL INFERENCE)
# ======================================================

def predict(res, repeat=20):

    transform = get_transform(res)

    img = Image.open(IMG_PATH).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # warm-up
    with torch.no_grad():
        model(input_tensor)

    start = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            output = model(input_tensor)

    end = time.time()

    avg_time = ((end - start) / repeat) * 1000  # ms

    prob = torch.softmax(output, dim=1)
    conf, pred = torch.max(prob, 1)

    return CLASS_NAMES[pred.item()], conf.item(), avg_time

# ======================================================
# 5. BANDWIDTH SCENARIOS
# ======================================================

def bandwidth_scenarios():
    return [
        ("Stable", 10.0),
        ("Medium", 5.0),
        ("Light Congestion", 3.0),
        ("Moderate", 1.5),
        ("Heavy", 0.5),
        ("Extreme", 0.2)
    ]

# ======================================================
# 6. ADAPTIVE RESOLUTION
# ======================================================

def adaptive_resolution(bw):

    if bw <= 1:
        return 64
    elif bw <= 2:
        return 96
    elif bw <= 5:
        return 112
    else:
        return 150

# ======================================================
# 7. LATENCY MODEL (REALISTIC)
# ======================================================

def compute_latency(res, bw, inf_time):

    # estimasi ukuran (MB)
    size_mb = (res * res * 3) / (1024 * 1024)

    # transmission delay (ms)
    transmission = (size_mb / bw) * 1000

    total_latency = transmission + inf_time

    return total_latency

# ======================================================
# 8. VISUAL FRAME
# ======================================================

def create_frame(res, label, conf, fps, latency, bw, mode):

    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_small = cv2.resize(img, (res,res))
    display = cv2.resize(img_small, (300,300), interpolation=cv2.INTER_NEAREST)

    overlay = display.copy()
    cv2.rectangle(overlay, (5,5), (260,130), (0,0,0), -1)
    cv2.addWeighted(overlay,0.6,display,0.4,0,display)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(display,f"Mode: {mode}",(10,20),font,0.5,(255,255,255),1)
    cv2.putText(display,f"Adaptive ON",(10,40),font,0.5,(255,200,0),1)
    cv2.putText(display,f"Res: {res}",(10,60),font,0.5,(255,255,255),1)
    cv2.putText(display,f"FPS: {int(fps)}",(10,80),font,0.5,(0,255,0),1)
    cv2.putText(display,f"Latency: {latency:.1f}ms",(10,100),font,0.5,(0,255,255),1)
    cv2.putText(display,f"BW: {bw} Mbps",(10,120),font,0.5,(200,200,200),1)

    cv2.putText(display,f"{label} ({conf*100:.1f}%)",(10,280),
                font,0.6,(0,255,0),2)

    return display

# ======================================================
# 9. MAIN SIMULATION
# ======================================================

frames = []
titles = []

for mode, bw in bandwidth_scenarios():

    res = adaptive_resolution(bw)

    label, conf, avg_time = predict(res)

    latency = compute_latency(res, bw, avg_time)

    fps = 1000 / (avg_time + 1e-6)

    frame = create_frame(res, label, conf, fps, latency, bw, mode)

    frames.append(frame)
    titles.append(f"{mode} ({res}px)")

# ======================================================
# 10. PLOT GRID
# ======================================================

fig, axes = plt.subplots(2,3, figsize=(15,10))
axes = axes.ravel()

for i in range(len(frames)):
    axes[i].imshow(frames[i])
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("adaptive_simulation_real.png", dpi=300)
plt.show()