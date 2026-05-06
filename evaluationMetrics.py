import matplotlib.pyplot as plt

# Data
resolutions = ["64×64", "96×96", "112×112", "150×150"]
accuracy = [0.843, 0.856, 0.855, 0.804]
precision = [0.85, 0.85, 0.85, 0.82]
recall = [0.85, 0.85, 0.84, 0.80]
f1 = [0.84, 0.84, 0.84, 0.80]
fps = [404, 244, 193, 72]
latency = [7.40, 12.25, 15.57, 24.10]
tradeoff = [0.2058, 0.1549, 0.0227, 0.0125]

# Create figure
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Table content
table_data = []
for i in range(len(resolutions)):
    table_data.append([
        resolutions[i],
        f"{accuracy[i]:.3f}",
        f"{precision[i]:.2f}",
        f"{recall[i]:.2f}",
        f"{f1[i]:.2f}",
        f"{fps[i]}",
        f"{latency[i]:.2f}",
        f"{tradeoff[i]:.4f}"
    ])

columns = ["Resolution", "Accuracy", "Precision", "Recall", "F1", "FPS", "Latency (ms)", "Trade-off"]

# Draw table
table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Highlight best values
# Accuracy best = 96×96
table[(2,1)].set_facecolor("#d4edda")
# FPS best = 64×64
table[(1,5)].set_facecolor("#d1ecf1")
# Trade-off best = 64×64
table[(1,7)].set_facecolor("#fff3cd")

plt.title("Performance Comparison Across Different Resolutions", fontsize=12)
plt.tight_layout()
plt.savefig("performance_table.png", dpi=300)
plt.show()