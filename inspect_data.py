import os
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"

files = [
    "competitions.csv",
    "matches.csv",
    "teams.csv",
    "playing_eleven.csv",
    "competition_team_player_mapping.csv",
    "ball_by_ball.csv"
]

print("=== DATASET INSPECTION ===")
for f in files:
    fpath = os.path.join(DATA_DIR, f)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"\n--- {f} ({size_mb:.2f} MB) ---")
        df_head = pd.read_csv(fpath, nrows=5)
        print("Columns:", list(df_head.columns))
        print("Shape preview:")
        print(df_head)
    else:
        print(f"Missing: {f}")
