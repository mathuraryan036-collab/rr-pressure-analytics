import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
CHARTS_DIR = os.path.join(DATA_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

bbb = pd.read_feather(os.path.join(DATA_DIR, "bbb_with_pressure.feather"))
import test_contextual_true_rates as tr

top10_batters = tr.top15_batters.head(10)
top10_bowlers = tr.top15_bowlers.head(10)
all_batters = tr.bat_cand
all_bowlers = tr.bowl_cand

# -------------------------------------------------------------
# CHART 1: Pressure Index Distribution across Match Phases
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
sns.kdeplot(data=bbb[bbb['innings'] == 1]['pressure_score'], ax=ax, label='1st Innings (Setting Target)', color='#2b5c8f', fill=True, alpha=0.3)
sns.kdeplot(data=bbb[bbb['innings'] == 2]['pressure_score'], ax=ax, label='2nd Innings (Chasing Target)', color='#e05638', fill=True, alpha=0.3)
ax.axvline(0.70, color='black', linestyle='--', linewidth=1.5, label='High-Pressure Cutoff (PI ≥ 0.70)')
ax.set_title('Pressure Index Distribution by Match Innings', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Pressure Index (0.0 = Routine Context, 1.0 = Peak Crunch)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(frameon=True, facecolor='white', loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "pressure_index_distribution.png"))
plt.close(fig)
print("Saved pressure_index_distribution.png")

# -------------------------------------------------------------
# CHART 2: Batting Clutch Quadrant (Pressure SR vs Baseline SR)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
ax.scatter(all_batters['sr_base'], all_batters['sr_press'], color='#cbd5e1', alpha=0.6, s=40, edgecolors='none', label='All Qualifying Batters (N=157)')

# Parity line
lims = [110, 215]
ax.plot(lims, lims, color='#94a3b8', linestyle=':', linewidth=1.5, label='Parity Line (No Change Under Pressure)')

# Highlight Top 10
ax.scatter(top10_batters['sr_base'], top10_batters['sr_press'], color='#2563eb', s=120, edgecolors='black', linewidth=1.2, zorder=5, label='Top 10 Pressure Batters')

for _, row in top10_batters.iterrows():
    ax.annotate(f"{row['Rank']}. {row['striker_name']}",
                (row['sr_base'], row['sr_press']),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, fontweight='bold', color='#1e293b')

ax.set_xlim(115, 185)
ax.set_ylim(130, 215)
ax.set_title('Batting Under Pressure: Pressure SR vs Baseline SR', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Baseline Strike Rate (Overall Context)', fontsize=12)
ax.set_ylabel('High-Pressure Strike Rate (PI ≥ 0.70)', fontsize=12)
ax.legend(frameon=True, facecolor='white', loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "batting_clutch_scatter.png"))
plt.close(fig)
print("Saved batting_clutch_scatter.png")

# -------------------------------------------------------------
# CHART 3: Bowling Clutch Quadrant (Pressure ER vs Baseline ER)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
ax.scatter(all_bowlers['er_base'], all_bowlers['er_press'], color='#cbd5e1', alpha=0.6, s=40, edgecolors='none', label='All Qualifying Bowlers (N=165)')

# Parity line
lims_b = [4.5, 12.0]
ax.plot(lims_b, lims_b, color='#94a3b8', linestyle=':', linewidth=1.5, label='Parity Line (No Change Under Pressure)')

# Global pressure average line
ax.axhline(tr.global_hp_er, color='#ef4444', linestyle='--', linewidth=1.2, label=f'Global Pressure Par ER ({tr.global_hp_er:.2f} rpo)')

# Highlight Top 10
ax.scatter(top10_bowlers['er_base'], top10_bowlers['er_press'], color='#059669', s=120, edgecolors='black', linewidth=1.2, zorder=5, label='Top 10 Pressure Bowlers')

for _, row in top10_bowlers.iterrows():
    ax.annotate(f"{row['Rank']}. {row['bowler_name']}",
                (row['er_base'], row['er_press']),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, fontweight='bold', color='#0f172a')

ax.set_xlim(5.0, 10.5)
ax.set_ylim(4.5, 11.5)
ax.set_title('Bowling Under Pressure: Pressure ER vs Baseline ER', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Baseline Economy Rate (Overall Context)', fontsize=12)
ax.set_ylabel('High-Pressure Economy Rate (PI ≥ 0.70)', fontsize=12)
ax.legend(frameon=True, facecolor='white', loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "bowling_clutch_scatter.png"))
plt.close(fig)
print("Saved bowling_clutch_scatter.png")

# -------------------------------------------------------------
# CHART 4: Top 10 Match-to-Match Consistency Comparison
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Batters
sns.barplot(data=top10_batters, y='striker_name', x='bat_consistency', ax=ax1, palette='Blues_r', edgecolor='black', linewidth=0.8)
ax1.set_title('Top 10 Batters: Match Consistency Rate', fontsize=12, fontweight='bold')
ax1.set_xlabel('Proportion of High-Pressure Matches Delivered (Success %)', fontsize=10)
ax1.set_ylabel('')
ax1.set_xlim(0, 1.0)
for p in ax1.patches:
    ax1.annotate(f"{p.get_width()*100:.1f}%", (p.get_width() + 0.02, p.get_y() + p.get_height()/2),
                 va='center', fontsize=9, fontweight='bold')

# Bowlers
sns.barplot(data=top10_bowlers, y='bowler_name', x='bowl_consistency', ax=ax2, palette='Greens_r', edgecolor='black', linewidth=0.8)
ax2.set_title('Top 10 Bowlers: Match Consistency Rate', fontsize=12, fontweight='bold')
ax2.set_xlabel('Proportion of High-Pressure Matches Delivered (Success %)', fontsize=10)
ax2.set_ylabel('')
ax2.set_xlim(0, 1.1)
for p in ax2.patches:
    ax2.annotate(f"{p.get_width()*100:.1f}%", (p.get_width() + 0.02, p.get_y() + p.get_height()/2),
                 va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "consistency_comparison.png"))
plt.close(fig)
print("Saved consistency_comparison.png")

print("All visual evidence generated successfully.")
