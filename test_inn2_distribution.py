import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))

bbb = bbb.merge(matches[['match_id', 'revised_target', 'win_team_id']], on='match_id', how='left')
bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

grouped = bbb.groupby(['match_id', 'innings'])
bbb['prev_cum_runs'] = grouped['total_ball_runs'].cumsum() - bbb['total_ball_runs']
bbb['prev_cum_wickets'] = grouped['is_wicket'].cumsum() - bbb['is_wicket']
bbb['prev_legal_balls'] = grouped['is_legal'].cumsum() - bbb['is_legal']

bbb['balls_remaining'] = np.maximum(0, 120 - bbb['prev_legal_balls'])
bbb['wickets_in_hand'] = np.maximum(0, 10 - bbb['prev_cum_wickets'])

inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1_totals.columns = ['match_id', 'inn1_total']
bbb = bbb.merge(inn1_totals, on='match_id', how='left')

bbb['target'] = np.where(bbb['revised_target'].notna() & (bbb['revised_target'] > 0),
                         bbb['revised_target'],
                         bbb['inn1_total'] + 1)

inn2 = bbb[bbb['innings'] == 2].copy()
inn2['runs_needed'] = np.maximum(0, inn2['target'] - inn2['prev_cum_runs'])
inn2['rrr'] = np.where(inn2['balls_remaining'] > 0, (inn2['runs_needed'] / inn2['balls_remaining']) * 6.0, 99.0)

print("Innings 2 sample percentiles of RRR:")
print(inn2['rrr'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
print("Innings 2 balls in overs 15-20:", len(inn2[inn2['overid'] >= 15]))
print("Innings 2 balls in overs 15-20 with RRR between 8 and 18:", len(inn2[(inn2['overid'] >= 15) & (inn2['rrr'] >= 8) & (inn2['rrr'] <= 18)]))
