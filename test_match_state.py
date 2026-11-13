import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))

# Sort
bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Merge metadata
m_meta = matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']]
bbb = bbb.merge(m_meta, on='match_id', how='left')

c_meta = competitions[['comp_id', 'comp_name', 'match_format', 'season']]
bbb = bbb.merge(c_meta, on='comp_id', how='left')

bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

# Compute cumulative runs and wickets prior to this delivery
# Group by match_id and innings
grouped = bbb.groupby(['match_id', 'innings'])

bbb['prev_cum_runs'] = grouped['total_ball_runs'].cumsum() - bbb['total_ball_runs']
bbb['prev_cum_wickets'] = grouped['is_wicket'].cumsum() - bbb['is_wicket']
bbb['prev_legal_balls'] = grouped['is_legal'].cumsum() - bbb['is_legal']

bbb['balls_remaining'] = np.maximum(0, 120 - bbb['prev_legal_balls'])
bbb['wickets_in_hand'] = np.maximum(1, 10 - bbb['prev_cum_wickets'])

# Calculate innings 1 total for each match
inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1_totals.columns = ['match_id', 'inn1_total']

bbb = bbb.merge(inn1_totals, on='match_id', how='left')

# Determine target for innings 2
# If revised_target is available and > 0, use it, else inn1_total + 1
bbb['target'] = np.where(bbb['revised_target'].notna() & (bbb['revised_target'] > 0),
                         bbb['revised_target'],
                         bbb['inn1_total'] + 1)

bbb['runs_needed'] = np.where(bbb['innings'] == 2,
                              np.maximum(0, bbb['target'] - bbb['prev_cum_runs']),
                              np.nan)

bbb['rrr'] = np.where((bbb['innings'] == 2) & (bbb['balls_remaining'] > 0),
                      (bbb['runs_needed'] / bbb['balls_remaining']) * 6.0,
                      np.nan)

print("Summary of Innings 2 state:")
inn2 = bbb[bbb['innings'] == 2]
print(inn2[['runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr']].describe())
