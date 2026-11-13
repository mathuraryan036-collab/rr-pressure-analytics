import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_feather(os.path.join(DATA_DIR, "bbb_with_pressure.feather"))
teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))

# Define competition quality weight
# IPL teams
ipl_teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Rajasthan Royals', 'Punjab Kings',
    'Royal Challengers Bengaluru', 'Delhi Capitals', 'Kolkata Knight Riders',
    'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
]
top_t20i_teams = ['India', 'Australia', 'England', 'South Africa', 'New Zealand', 'Pakistan', 'West Indies', 'Afghanistan']

# Match quality weight
is_ipl = bbb['comp_name'].str.contains('IPL', case=False, na=False)

bbb['comp_weight'] = np.where(is_ipl, 1.25,
                     np.where(bbb['match_format'] == 'T20I', 1.0, 0.85))

print("Competition weight distribution:")
print(bbb['comp_weight'].value_counts())

# Check how many batters have >= 100, >= 120, >= 150 balls under high pressure
bbb['batter_ball_faced'] = bbb['extras_type'] != 'wide'
bat_counts = bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']].groupby(['striker_id', 'striker_name']).size()
print(f"\nBatters with >= 75 high-pressure balls: {(bat_counts >= 75).sum()}")
print(f"Batters with >= 100 high-pressure balls: {(bat_counts >= 100).sum()}")
print(f"Batters with >= 120 high-pressure balls: {(bat_counts >= 120).sum()}")
print(f"Batters with >= 150 high-pressure balls: {(bat_counts >= 150).sum()}")

# Check bowlers
bowl_counts = bbb[bbb['is_legal'] & bbb['is_high_pressure']].groupby(['bowler_id', 'bowler_name']).size()
print(f"\nBowlers with >= 75 high-pressure balls: {(bowl_counts >= 75).sum()}")
print(f"Bowlers with >= 100 high-pressure balls: {(bowl_counts >= 100).sum()}")
print(f"Bowlers with >= 120 high-pressure balls: {(bowl_counts >= 120).sum()}")
print(f"Bowlers with >= 150 high-pressure balls: {(bowl_counts >= 150).sum()}")
