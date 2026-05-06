import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("final_system_test_cpu.csv")

# urutkan berdasarkan resolusi
df = df.sort_values("Resolution")

# ==============================
# 1. ACCURACY vs RESOLUTION
# ==============================

plt.figure()
plt.plot(df["Resolution"], df["Accuracy"], marker='o')
plt.title("Accuracy vs Resolution")
plt.xlabel("Resolution (pixels)")
plt.ylabel("Accuracy")
plt.grid(True)

for x, y in zip(df["Resolution"], df["Accuracy"]):
    plt.text(x, y, f"{y:.3f}", ha='center', va='bottom')

plt.savefig("accuracy_vs_resolution.png", dpi=300)
plt.show()

# ==============================
# 2. FPS vs RESOLUTION
# ==============================

plt.figure()
plt.plot(df["Resolution"], df["FPS"], marker='o')
plt.title("FPS vs Resolution")
plt.xlabel("Resolution (pixels)")
plt.ylabel("Frames Per Second (FPS)")
plt.grid(True)

for x, y in zip(df["Resolution"], df["FPS"]):
    plt.text(x, y, f"{int(y)}", ha='center', va='bottom')

plt.savefig("fps_vs_resolution.png", dpi=300)
plt.show()

# ==============================
# 3. LATENCY vs RESOLUTION
# ==============================

plt.figure()
plt.plot(df["Resolution"], df["Latency"], marker='o')
plt.title("Latency vs Resolution")
plt.xlabel("Resolution (pixels)")
plt.ylabel("Latency (ms)")
plt.grid(True)

for x, y in zip(df["Resolution"], df["Latency"]):
    plt.text(x, y, f"{y:.1f}ms", ha='center', va='bottom')

plt.savefig("latency_vs_resolution.png", dpi=300)
plt.show()