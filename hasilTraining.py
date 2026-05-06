import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data dari CSV
file_path = "final_system_test_cpu.csv"
try:
    df = pd.read_csv(file_path)
    # Pastikan data diurutkan berdasarkan resolusi untuk grafik yang rapi
    df = df.sort_values('Resolution')
except FileNotFoundError:
    print("File CSV tidak ditemukan!")
    exit()

# Set style visualisasi
sns.set_theme(style="whitegrid")

# 2. Membuat Subplots (2 Grafik dalam 1 Gambar)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- GRAFIK 1: Inference Time vs Resolution ---
sns.lineplot(data=df, x='Resolution', y='InferenceTime', marker='o', ax=ax1, color='red', linewidth=2)
ax1.set_title('Inference Latency per Resolution', fontsize=14)
ax1.set_xlabel('Resolution (pixels)', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)
for x, y in zip(df['Resolution'], df['InferenceTime']):
    ax1.text(x, y + 0.5, f'{y:.2f}ms', ha='center', va='bottom', fontsize=10)

# --- GRAFIK 2: FPS vs Resolution ---
sns.barplot(data=df, x='Resolution', y='FPS', ax=ax2, palette='viridis')
ax2.set_title('Throughput (FPS) per Resolution', fontsize=14)
ax2.set_xlabel('Resolution (pixels)', fontsize=12)
ax2.set_ylabel('Frames Per Second', fontsize=12)
for i, v in enumerate(df['FPS']):
    ax2.text(i, v + 2, f'{int(v)}', ha='center', va='bottom', fontweight='bold')

# 3. Final Touch & Save
plt.tight_layout()
plt.savefig('latency_fps_analysis.png', dpi=300)
print("Grafik berhasil disimpan sebagai 'latency_fps_analysis.png'")
plt.show()

# 4. Tambahan: Analisis Trade-off (Akurasi vs Latency)
plt.figure(figsize=(8, 6))
plt.scatter(df['InferenceTime'], df['Accuracy'], s=100, color='purple')
for i, txt in enumerate(df['Resolution']):
    plt.annotate(f"{txt}px", (df['InferenceTime'].iat[i], df['Accuracy'].iat[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Accuracy vs. Latency Trade-off', fontsize=14)
plt.xlabel('Inference Latency (ms)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('accuracy_latency_tradeoff.png', dpi=300)
plt.show()