import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
CHARTS_DIR = os.path.join(DATA_DIR, "charts")

# Load existing charts
img_dist = mpimg.imread(os.path.join(CHARTS_DIR, "pressure_index_distribution.png"))
img_bat = mpimg.imread(os.path.join(CHARTS_DIR, "batting_clutch_scatter.png"))
img_bowl = mpimg.imread(os.path.join(CHARTS_DIR, "bowling_clutch_scatter.png"))
img_const = mpimg.imread(os.path.join(CHARTS_DIR, "consistency_comparison.png"))

# Create 2x2 Composite Showcase Poster
fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)

axes[0, 0].imshow(img_dist)
axes[0, 0].axis('off')
axes[0, 0].set_title("Step 1: Quantifying Pressure (200k+ Deliveries)", fontsize=16, fontweight='bold', pad=10)

axes[0, 1].imshow(img_bat)
axes[0, 1].axis('off')
axes[0, 1].set_title("Step 2: Batting Clutch Quadrant (SR Elevation)", fontsize=16, fontweight='bold', pad=10)

axes[1, 0].imshow(img_bowl)
axes[1, 0].axis('off')
axes[1, 0].set_title("Step 3: Bowling Run Suppression (True ER Saved)", fontsize=16, fontweight='bold', pad=10)

axes[1, 1].imshow(img_const)
axes[1, 1].axis('off')
axes[1, 1].set_title("Step 4: Repeat Reliability Across Matches", fontsize=16, fontweight='bold', pad=10)

plt.suptitle("Rajasthan Royals SupeRR Selector S2: Analytical Journey & System Architecture", fontsize=22, fontweight='bold', y=0.98)
plt.tight_layout()
fig.subplots_adjust(top=0.93)

out_path = os.path.join(CHARTS_DIR, "journey_showcase_poster.png")
fig.savefig(out_path)
plt.close(fig)

# Also copy to artifact dir
import shutil
shutil.copy(out_path, r"C:\Users\ASUS\.gemini\antigravity\brain\5ee2fa8d-ed75-4925-a270-ee5ca341ef13\charts\journey_showcase_poster.png")
print("Saved journey_showcase_poster.png successfully.")
