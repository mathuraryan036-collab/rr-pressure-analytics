import os
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))

# Check a sample match
sample_m = bbb[bbb['match_id'] == 13511]
print("Innings in match 13511:", sample_m['innings'].unique())
for inn in sample_m['innings'].unique():
    inn_df = sample_m[sample_m['innings'] == inn]
    print(f"Inn {inn} balls: {len(inn_df)}, total runs off bat: {inn_df['runs'].sum()}, extras1: {inn_df['extras1_runs'].sum()}, extras2: {inn_df['extras_2_runs'].sum()}")

# Check revised_target in matches.csv
print("\nRevised target sample in matches.csv:")
print(matches[matches['revised_target'].notna()][['match_id', 'revised_target', 'first_inn_revised_overs', 'second_inn_revised_overs']].head())
