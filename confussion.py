import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================================================
# 1. DEFINISI ARSITEKTUR MODEL (Wajib Sama dengan Training)
# ======================================================
class AdaptiveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6) 
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ======================================================
# 2. KONFIGURASI
# ======================================================
DEVICE = torch.device("cpu")
TEST_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test"
MODEL_FILE = "best_model_cpu.pth"
RESOLUTIONS = [150, 112, 96, 64]  # Daftar semua resolusi yang diminta
BATCH_SIZE = 8

def generate_all_confusion_matrices():
    # Load model sekali saja
    model = AdaptiveCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        model.eval()
        print("Model berhasil dimuat.")
    except:
        print("Gagal memuat model. Pastikan file .pth ada di folder yang sama.")
        return

    # Loop untuk setiap resolusi
    for res in RESOLUTIONS:
        print(f"\n--- Memproses Resolusi {res}x{res} ---")
        
        # Transformasi sesuai resolusi saat ini
        transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = datasets.ImageFolder(TEST_PATH, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        class_names = test_dataset.classes

        y_true = []
        y_pred = []

        # Prediksi
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Hitung Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Visualisasi & Simpan Gambar
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - Resolution {res}px')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Simpan file gambar otomatis (cm_150px.png, cm_112px.png, dst)
        filename = f"confusion_matrix_{res}px.png"
        plt.savefig(filename, dpi=300)
        print(f"Gambar disimpan sebagai: {filename}")
        plt.show()

        # Print Report Singkat ke Konsol
        print(f"Report for {res}px:")
        print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    generate_all_confusion_matrices()