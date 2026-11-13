import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))

print(f"Loaded {len(bbb)} balls across {bbb['match_id'].nunique()} matches.")

# Sort chronologically by match_id, innings, overid, over_ball_num
bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Merge match metadata
m_meta = matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']]
bbb = bbb.merge(m_meta, on='match_id', how='left')

# Merge competition metadata
c_meta = competitions[['comp_id', 'comp_name', 'match_format', 'season']]
bbb = bbb.merge(c_meta, on='comp_id', how='left')

# Define legal balls
bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])

# Calculate total runs per ball (batter runs + extras)
# extras1_runs and extras_2_runs
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

# Bowler runs conceded: runs off bat + wide + no ball (not byes / legbyes)
is_bye_legbye = bbb['extras_type'].isin(['byes', 'legbyes', 'Penalty'])
bbb['bowler_runs_conceded'] = bbb['runs'] + np.where(is_bye_legbye, 0, bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0))

# Bowler wicket: dismissal_type in caught, bowled, lbw, stump, hit wicket
bowler_dismissals = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']
bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(bowler_dismissals)

print("Computed ball basics successfully.")
