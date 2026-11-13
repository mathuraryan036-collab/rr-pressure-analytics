import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"

competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"), nrows=50000)

print("=== COMPETITIONS SUMMARY ===")
print("Competitions count:", len(competitions))
print("Seasons:", competitions['season'].value_counts())
print("Match formats:", competitions['match_format'].value_counts())
print("\nSample competitions:")
print(competitions[competitions['comp_name'].str.contains('IPL|Premier League|Indian', case=False, na=False)][['comp_id', 'comp_name', 'season']].head(10))

print("\n=== MATCHES SUMMARY ===")
print("Total matches in matches.csv:", len(matches))
matches['year'] = pd.to_datetime(matches['match_date']).dt.year
print("Matches per year:", matches['year'].value_counts().sort_index())
print("Knockout values:", matches['knock_out'].value_counts(dropna=False))

print("\n=== BALL BY BALL (First 50k rows) ===")
print("Over IDs min/max:", bbb['overid'].min(), bbb['overid'].max())
print("Over ball num min/max:", bbb['over_ball_num'].min(), bbb['over_ball_num'].max())
print("Innings:", bbb['innings'].value_counts())
print("Runs distribution:\n", bbb['runs'].value_counts().head(10))
print("Extras type:\n", bbb['extras_type'].value_counts(dropna=False))
print("Extras 1 runs:\n", bbb['extras1_runs'].value_counts().head(5))
print("Extras 2 runs:\n", bbb['extras_2_runs'].value_counts().head(5))
print("Dismissal types:\n", bbb['dismissal_type'].value_counts(dropna=False))

# Check how total runs per ball should be computed
sample_extras = bbb[bbb['extras_type'].notna()][['runs', 'extras_type', 'extras1_runs', 'extras_2_runs', 'is_four', 'is_six', 'is_wicket']].head(10)
print("\nSample extras rows:")
print(sample_extras)
