import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))

print("Top 20 competitions by match count:")
comp_counts = matches['comp_id'].value_counts().reset_index()
comp_counts.columns = ['comp_id', 'match_count']
comp_counts = comp_counts.merge(competitions, on='comp_id', how='left')
print(comp_counts.head(20)[['comp_id', 'comp_name', 'match_format', 'season', 'match_count']])

# Check all teams
print(f"\nTotal unique teams: {teams['team_name'].nunique()}")
print("Sample teams:")
print(teams['team_name'].drop_duplicates().head(30).tolist())
